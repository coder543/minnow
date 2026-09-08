use super::*;

/// One independent sequence in a transformer invocation. Each segment consists
/// of complete diffusion blocks; attention and positions remain sequence-local.
pub struct Forward<'a> {
    pub tokens: &'a [u32],
    pub cache: &'a mut Cache,
    pub logits: bool,
}

impl Model {
    pub fn forward_batch(&self, batches: &mut [Forward<'_>]) -> Result<Vec<Option<Tensor>>> {
        ensure!(!batches.is_empty(), "empty inference batch");
        if batches.len() == 1 {
            let b = &mut batches[0];
            return Ok(vec![self.forward(b.tokens, b.cache, b.logits, None)?]);
        }
        let c = &self.config;
        let total = batches
            .iter()
            .try_fold(0usize, |n, b| n.checked_add(b.tokens.len()))
            .context("batch size overflow")?;
        ensure!(
            total <= MAX_FORWARD_TOKENS,
            "batch exceeds transformer workspace limit"
        );
        for batch in batches.iter() {
            ensure!(
                !batch.tokens.is_empty()
                    && batch.tokens.len().is_multiple_of(c.block_size)
                    && batch.cache.len.is_multiple_of(c.block_size)
                    && batch.cache.layers.len() == c.num_hidden_layers
                    && batch.cache.len + batch.tokens.len() <= batch.cache.capacity,
                "invalid sequence blocks or cache capacity in batch"
            );
            ensure!(
                batch.tokens.iter().all(|&t| (t as usize) < c.vocab_size),
                "token outside vocabulary"
            );
        }
        let tokens: Vec<u32> = batches
            .iter()
            .flat_map(|b| b.tokens.iter().copied())
            .collect();
        let mut x = self
            .embeddings
            .index_select(&Tensor::from_vec(tokens, total, &self.device)?, 0)?;
        let mut ropes = Vec::new();
        for batch in batches.iter_mut() {
            let cache = &mut *batch.cache;
            cache.staged_tokens = None;
            let (offset, n) = (cache.len, batch.tokens.len());
            if !matches!(&cache.rope,Some((p,count,_,_)) if *p==offset && *count==n) {
                let (cos, sin) = self.rope(n, offset)?;
                cache.rope = Some((offset, n, cos, sin));
            }
            let (_, _, cos, sin) = cache.rope.as_ref().unwrap();
            ropes.push((cos.clone(), sin.clone()));
        }
        for (i, layer) in self.layers.iter().enumerate() {
            self.refresh_workspace_cache()?;
            let h = layer.input_norm.forward(&x)?;
            let qkv = layer.qkv.forward(&h)?.reshape((
                total,
                c.num_attention_heads + 2 * c.num_key_value_heads,
                c.head_dim,
            ))?;
            let mut attention = Vec::new();
            let mut row = 0;
            for (batch, (cos, sin)) in batches.iter_mut().zip(&ropes) {
                let cache = &mut *batch.cache;
                let n = batch.tokens.len();
                let (q, k, v) = self.prepare_qkv(layer, &qkv.narrow(0, row, n)?, cos, sin)?;
                if cache.layers[i].is_none() {
                    let shape = (c.num_key_value_heads, cache.capacity, c.head_dim);
                    cache.layers[i] = Some((
                        Tensor::zeros(shape, self.dtype, &self.device)?,
                        Tensor::zeros(shape, self.dtype, &self.device)?,
                    ));
                }
                let (keys, values) = cache.layers[i].as_ref().unwrap();
                keys.slice_set(&k, 1, cache.len)?;
                values.slice_set(&v, 1, cache.len)?;
                attention.push(self.attention(&q, keys, values, cache.len)?);
                row += n;
            }
            x = (x + layer.out.forward(&Tensor::cat(&attention, 0)?)?)?;
            let h = layer.post_norm.forward(&x)?;
            let y = match &layer.mlp {
                FeedForward::Dense(mlp) => mlp.forward(&h, self.moe_execution.fused_activation)?,
                // Complete block alignment prevents router capacity from mixing
                // tokens belonging to different requests.
                FeedForward::Sparse(moe) => {
                    moe.forward(&h, c, &mut None, "", &self.moe_execution)?
                }
            };
            x = (x + y)?;
        }
        let mut output = vec![None; batches.len()];
        let mut selected = Vec::new();
        let mut row = 0;
        for batch in batches.iter() {
            if batch.logits {
                selected.push(x.narrow(0, row, batch.tokens.len())?);
            }
            row += batch.tokens.len();
        }
        if !selected.is_empty() {
            let h = self.norm.forward(&Tensor::cat(&selected, 0)?)?;
            let logits = self.head.forward(&h)?.to_dtype(DType::F32)?;
            let mut row = 0;
            for (batch, out) in batches.iter().zip(&mut output) {
                if batch.logits {
                    *out = Some(logits.narrow(0, row, batch.tokens.len())?);
                    row += batch.tokens.len();
                }
            }
        }
        for batch in batches.iter_mut() {
            batch.cache.staged_tokens = Some(batch.tokens.to_vec());
            batch.cache.forwards += 1;
            batch.cache.processed_tokens += batch.tokens.len();
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ragged_batches_preserve_positions_attention_boundaries_and_commit_state() -> Result<()> {
        let model = Model::load(Path::new("tests/fixtures/tiny"), DType::F32, &Device::Cpu)?;
        let mut a = Cache::new(&model.config, 128)?;
        let mut b = Cache::new(&model.config, 192)?;
        crate::decode::prefill(&model, &[7; 64], &mut b, || false)?;
        let a_tokens: Vec<u32> = (0..64).map(|i| i % 19 + 1).collect();
        let b_tokens = vec![23; 32];
        let mut a_expected = Cache::new(&model.config, 128)?;
        let mut b_expected = Cache::new(&model.config, 192)?;
        crate::decode::prefill(&model, &[7; 64], &mut b_expected, || false)?;
        let expected_a = model
            .forward(&a_tokens, &mut a_expected, true, None)?
            .unwrap();
        let expected_b = model
            .forward(&b_tokens, &mut b_expected, true, None)?
            .unwrap();
        let actual = model.forward_batch(&mut [
            Forward {
                tokens: &a_tokens,
                cache: &mut a,
                logits: false,
            },
            Forward {
                tokens: &b_tokens,
                cache: &mut b,
                logits: true,
            },
        ])?;
        assert!(actual[0].is_none());
        let error = (&expected_b - actual[1].as_ref().unwrap())?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            error < 1e-5,
            "cross-sequence attention or position error: {error}"
        );
        a.commit(&a_tokens)?;
        b.commit(&b_tokens)?;
        assert_eq!((a.len(), b.len()), (64, 96));
        assert_eq!((a.forwards, b.forwards), (1, 2));
        a_expected.commit(&a_tokens)?;
        b_expected.commit(&b_tokens)?;
        let next = model.forward_batch(&mut [
            Forward {
                tokens: &[3; 32],
                cache: &mut a,
                logits: true,
            },
            Forward {
                tokens: &[5; 32],
                cache: &mut b,
                logits: true,
            },
        ])?;
        for (actual, tokens, cache) in [
            (next[0].as_ref().unwrap(), vec![3; 32], &mut a_expected),
            (next[1].as_ref().unwrap(), vec![5; 32], &mut b_expected),
        ] {
            let expected = model.forward(&tokens, cache, true, None)?.unwrap();
            assert!((actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-5);
        }
        // First segment's activations were computed even though no logits were requested.
        assert!(expected_a.abs()?.max_all()?.to_scalar::<f32>()? > 0.);
        Ok(())
    }
}
