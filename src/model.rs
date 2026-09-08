use crate::config::Config;
use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
#[cfg(feature = "cuda")]
use std::collections::BTreeSet;
use std::{collections::HashMap, path::Path};

struct Norm {
    weight: Tensor,
    eps: f64,
    fused: bool,
}
impl Norm {
    fn load(size: usize, eps: f64, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
            fused: true,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.fused && x.device().is_cuda() && x.dtype() == DType::BF16 {
            return Ok(crate::cuda::rms_norm(x, &self.weight, self.eps)?);
        }
        // Preserve the reference's cast BEFORE the learned scale multiplication.
        let f = x.to_dtype(DType::F32)?;
        let inv = (f.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?
            .sqrt()?
            .recip()?;
        Ok(f.broadcast_mul(&inv)?
            .to_dtype(x.dtype())?
            .broadcast_mul(&self.weight)?)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}
fn silu_mul(gate: &Tensor, up: &Tensor, _fused: bool) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if _fused && gate.device().is_cuda() && gate.dtype() == DType::BF16 {
        return Ok(crate::cuda::silu_mul(gate, up)?);
    }
    let activated = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?.to_dtype(gate.dtype())?;
    Ok((activated * up)?)
}
impl Mlp {
    fn load(h: usize, i: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            gate: candle_nn::linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up: candle_nn::linear_no_bias(h, i, vb.pp("up_proj"))?,
            down: candle_nn::linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }
    fn forward(&self, x: &Tensor, fused: bool) -> Result<Tensor> {
        let gate = self.gate.forward(x)?;
        Ok(self
            .down
            .forward(&silu_mul(&gate, &self.up.forward(x)?, fused)?)?)
    }
}

struct Moe {
    gate: Linear,
    bias: Vec<f32>,
    #[cfg(feature = "cuda")]
    device_bias: Tensor,
    experts: Vec<Mlp>,
    shared: Option<Mlp>,
    #[cfg(feature = "cuda")]
    packed: Vec<Tensor>,
}

struct MoeExecution {
    batched_experts: bool,
    compact_decode_experts: bool,
    fused_activation: bool,
    fused_mix: bool,
    device_routing: bool,
}

fn mix_experts(
    out: &Tensor,
    rows: Vec<u32>,
    weights: Vec<f32>,
    top_k: usize,
    _fused: bool,
) -> Result<Tensor> {
    let dtype = out.dtype();
    #[cfg(feature = "cuda")]
    if _fused && out.device().is_cuda() && out.dtype() == DType::BF16 && top_k <= 32 {
        return Ok(crate::cuda::mix_experts(out, &rows, &weights, top_k)?);
    }
    let tokens = rows.len() / top_k;
    let out = out
        .index_select(&Tensor::from_vec(rows, tokens * top_k, out.device())?, 0)?
        .reshape((tokens, top_k, out.dim(1)?))?
        .to_dtype(DType::F32)?;
    let weights = Tensor::from_vec(weights, (tokens, top_k, 1), out.device())?;
    Ok(out.broadcast_mul(&weights)?.sum(1)?.to_dtype(dtype)?)
}

/// Selection uses biased scores; mixing uses unbiased scores. Blocks never mix.
pub fn route(scores: &[Vec<f32>], bias: &[f32], c: &Config) -> Result<(Vec<u32>, Vec<f32>)> {
    ensure!(
        scores.len().is_multiple_of(c.block_size)
            && scores.iter().all(|s| s.len() == c.num_experts)
            && bias.len() == c.num_experts,
        "invalid routing shape"
    );
    let mut ids = Vec::with_capacity(scores.len() * c.num_experts_per_tok);
    let mut weights = Vec::with_capacity(ids.capacity());
    for block in scores.chunks_exact(c.block_size) {
        let maxima: Vec<f32> = (0..c.num_experts)
            .map(|e| {
                block
                    .iter()
                    .map(|s| s[e] + bias[e])
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        let mut allowed: Vec<usize> = (0..c.num_experts).collect();
        allowed.sort_unstable_by(|&a, &b| maxima[b].total_cmp(&maxima[a]).then(a.cmp(&b)));
        allowed.truncate(c.expert_capacity);
        for scores in block {
            allowed.sort_unstable_by(|&a, &b| {
                (scores[b] + bias[b])
                    .total_cmp(&(scores[a] + bias[a]))
                    .then(a.cmp(&b))
            });
            let selected = &allowed[..c.num_experts_per_tok];
            let sum = if c.num_experts_per_tok > 1 {
                selected.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20
            } else {
                1.0
            };
            for &e in selected {
                ids.push(e as u32);
                weights.push(scores[e] / sum * c.routed_scaling_factor as f32);
            }
        }
    }
    Ok((ids, weights))
}

impl Moe {
    fn load(c: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        // The streaming backend writes experts into their final packed storage.
        let mut packed = Vec::new();
        for (name, shape) in [
            ("gate_proj", (c.moe_intermediate_size, c.hidden_size)),
            ("up_proj", (c.moe_intermediate_size, c.hidden_size)),
            ("down_proj", (c.hidden_size, c.moe_intermediate_size)),
        ] {
            packed.push(vb.get(
                (c.num_experts, shape.0, shape.1),
                &format!("experts.{name}.weight"),
            )?);
        }
        let experts = (0..c.num_experts)
            .map(|e| {
                Ok(Mlp {
                    gate: Linear::new(packed[0].get(e)?, None),
                    up: Linear::new(packed[1].get(e)?, None),
                    down: Linear::new(packed[2].get(e)?, None),
                })
            })
            .collect::<Result<_>>()?;
        let bias = vb
            .clone()
            .set_device(Device::Cpu)
            .to_dtype(DType::F32)
            .get(c.num_experts, "gate.expert_bias")?
            .to_vec1::<f32>()?;
        Ok(Self {
            gate: Linear::new(
                vb.to_dtype(DType::F32)
                    .get((c.num_experts, c.hidden_size), "gate.weight")?,
                None,
            ),
            #[cfg(feature = "cuda")]
            device_bias: Tensor::from_slice(&bias, c.num_experts, vb.device())?,
            bias,
            experts,
            #[cfg(feature = "cuda")]
            packed,
            shared: if c.num_shared_experts > 0 {
                Some(Mlp::load(
                    c.hidden_size,
                    c.moe_intermediate_size * c.num_shared_experts,
                    vb.pp("shared_experts"),
                )?)
            } else {
                None
            },
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        c: &Config,
        trace: &mut Option<&mut Trace>,
        name: &str,
        execution: &MoeExecution,
    ) -> Result<Tensor> {
        let fused = execution.fused_activation;
        let logits = self.gate.forward(&x.to_dtype(DType::F32)?)?;
        record(trace, format!("{name}.router_logits"), &logits);
        #[cfg(feature = "cuda")]
        if execution.device_routing
            && execution.batched_experts
            && execution.fused_mix
            && x.device().is_cuda()
            && x.dtype() == DType::BF16
            && x.dim(0)? == 32
            && c.block_size == 32
            && c.num_experts == 256
            && c.num_experts_per_tok == 8
            && c.expert_capacity == 48
        {
            let plan = crate::cuda::route_mini(
                &logits,
                &self.device_bias,
                c.routed_scaling_factor as f32,
            )?;
            if trace.is_some() {
                record(trace, format!("{name}.router_ids"), &plan.expert_ids()?);
            }
            let gate = crate::cuda::routed_expert_gemm(x, &self.packed[0], &plan, false)?;
            let up = crate::cuda::routed_expert_gemm(x, &self.packed[1], &plan, false)?;
            let hidden = silu_mul(&gate, &up, fused)?;
            let experts = crate::cuda::routed_expert_gemm(&hidden, &self.packed[2], &plan, true)?;
            let mut out = crate::cuda::mix_routed_experts(&experts, &plan)?;
            if let Some(shared) = &self.shared {
                out = (out + shared.forward(x, fused)?)?;
            }
            return Ok(out);
        }
        let scores = candle_nn::ops::sigmoid(&logits)?.to_vec2::<f32>()?;
        let (ids, weights) = route(&scores, &self.bias, c)?;
        if trace.is_some() {
            record(
                trace,
                format!("{name}.router_ids"),
                &Tensor::from_vec(ids.clone(), (x.dim(0)?, c.num_experts_per_tok), x.device())?,
            );
        }
        // Batch each distinct expert once, with all block rows sharing its weights.
        #[cfg(feature = "cuda")]
        if execution.batched_experts
            && !execution.compact_decode_experts
            && x.device().is_cuda()
            && x.dtype() == DType::BF16
            && x.dim(0)? == c.block_size
        {
            let selected: Vec<usize> = ids
                .iter()
                .map(|&e| e as usize)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let gate = crate::cuda::expert_gemm(x, &self.packed[0], &selected, false)?;
            let up = crate::cuda::expert_gemm(x, &self.packed[1], &selected, false)?;
            let hidden = silu_mul(&gate, &up, fused)?;
            let out = crate::cuda::expert_gemm(&hidden, &self.packed[2], &selected, true)?
                .flatten_to(1)?;
            let mut mapping = vec![0; c.num_experts];
            for (i, &e) in selected.iter().enumerate() {
                mapping[e] = i;
            }
            let rows: Vec<u32> = ids
                .iter()
                .enumerate()
                .map(|(slot, &e)| {
                    (mapping[e as usize] * c.block_size + slot / c.num_experts_per_tok) as u32
                })
                .collect();
            let mut out = mix_experts(
                &out,
                rows,
                weights,
                c.num_experts_per_tok,
                execution.fused_mix,
            )?;
            if let Some(shared) = &self.shared {
                out = (out + shared.forward(x, fused)?)?;
            }
            return Ok(out);
        }
        let mut assignments = vec![Vec::new(); c.num_experts];
        for (slot, &e) in ids.iter().enumerate() {
            assignments[e as usize].push(slot);
        }
        let mut inverse = vec![0u32; ids.len()];
        let mut token_ids = Vec::with_capacity(ids.len());
        let mut segments = Vec::new();
        for (e, slots) in assignments
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.is_empty())
        {
            segments.push((e, slots.len()));
            for &slot in slots {
                inverse[slot] = token_ids.len() as u32;
                token_ids.push((slot / c.num_experts_per_tok) as u32);
            }
        }
        let selected = x.index_select(&Tensor::from_vec(token_ids, ids.len(), x.device())?, 0)?;
        let evaluate_serial = || -> Result<Tensor> {
            let mut outputs = Vec::new();
            let mut row = 0;
            for &(e, n) in &segments {
                outputs.push(self.experts[e].forward(&selected.narrow(0, row, n)?, fused)?);
                row += n;
            }
            Ok(Tensor::cat(&outputs, 0)?)
        };
        #[cfg(feature = "cuda")]
        let outputs =
            if execution.batched_experts && x.device().is_cuda() && x.dtype() == DType::BF16 {
                let gate = crate::cuda::grouped_expert_gemm(&selected, &self.packed[0], &segments)?;
                let up = crate::cuda::grouped_expert_gemm(&selected, &self.packed[1], &segments)?;
                let hidden = silu_mul(&gate, &up, fused)?;
                crate::cuda::grouped_expert_gemm(&hidden, &self.packed[2], &segments)?
            } else {
                evaluate_serial()?
            };
        #[cfg(not(feature = "cuda"))]
        let outputs = evaluate_serial()?;
        let mut y = mix_experts(
            &outputs,
            inverse,
            weights,
            c.num_experts_per_tok,
            execution.fused_mix,
        )?;
        if let Some(shared) = &self.shared {
            y = (y + shared.forward(x, fused)?)?;
        }
        Ok(y)
    }
}

enum FeedForward {
    Dense(Mlp),
    Sparse(Moe),
}
struct Layer {
    input_norm: Norm,
    post_norm: Norm,
    qkv: Linear,
    out: Linear,
    qnorm: Option<Norm>,
    knorm: Option<Norm>,
    mlp: FeedForward,
}

/// Scratch slots after `len` may change on every forward. Only `commit` advances it.
pub struct Cache {
    layers: Vec<Option<(Tensor, Tensor)>>,
    len: usize,
    capacity: usize,
    staged_tokens: Option<Vec<u32>>,
    rope: Option<(usize, usize, Tensor, Tensor)>,
    pub forwards: usize,
    pub processed_tokens: usize,
}
impl Cache {
    pub fn new(c: &Config, capacity: usize) -> Result<Self> {
        ensure!(
            capacity > 0
                && capacity <= c.max_position_embeddings
                && capacity.is_multiple_of(c.block_size),
            "cache capacity must be a positive whole number of blocks within model context"
        );
        Ok(Self {
            layers: vec![None; c.num_hidden_layers],
            len: 0,
            capacity,
            staged_tokens: None,
            rope: None,
            forwards: 0,
            processed_tokens: 0,
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub(crate) fn staged_matches(&self, tokens: &[u32]) -> bool {
        self.staged_tokens.as_deref() == Some(tokens)
    }
    pub fn commit(&mut self, tokens: &[u32]) -> Result<()> {
        ensure!(
            self.staged_matches(tokens),
            "cache commit requires a successful forward of the exact final tokens"
        );
        self.len += tokens.len();
        self.staged_tokens = None;
        Ok(())
    }
}

pub type Trace = HashMap<String, Tensor>;

// Bound each attention tile independently of the transformer prefill batch.
pub const MAX_ATTENTION_ELEMENTS: usize = 128 * 1024 * 1024;
// Also bound token/expert activations independently of the attention tile.
pub const MAX_FORWARD_TOKENS: usize = 8192;
fn record(trace: &mut Option<&mut Trace>, name: String, tensor: &Tensor) {
    if let Some(t) = trace {
        t.insert(name, tensor.clone());
    }
}

pub struct Model {
    pub config: Config,
    embeddings: Tensor,
    layers: Vec<Layer>,
    norm: Norm,
    head: Linear,
    device: Device,
    dtype: DType,
    moe_execution: MoeExecution,
    fused_attention: bool,
    fused_qkv: bool,
    prefill_chunk_tokens: usize,
    attention_chunk_tokens: usize,
    _lease: Option<crate::weights::ModelLease>,
}
impl Model {
    pub fn load(path: &Path, dtype: DType, device: &Device) -> Result<Self> {
        let c = Config::load(path)?;
        let loader = crate::weights::WeightLoader::open(path)?;
        let lease = loader.lease_and_check(dtype)?;
        let vb = VarBuilder::from_backend(Box::new(loader), dtype, device.clone());
        let embeddings = vb.get(
            (c.vocab_size, c.hidden_size),
            "model.word_embeddings.weight",
        )?;
        let mut layers = Vec::new();
        for i in 0..c.num_hidden_layers {
            tracing::info!(layer = i, total = c.num_hidden_layers, "loading layer");
            let v = vb.pp(format!("model.layers.{i}"));
            layers.push(Layer {
                input_norm: Norm::load(c.hidden_size, c.rms_norm_eps, v.pp("input_layernorm"))?,
                post_norm: Norm::load(
                    c.hidden_size,
                    c.rms_norm_eps,
                    v.pp("post_attention_layernorm"),
                )?,
                qkv: candle_nn::linear_no_bias(
                    c.hidden_size,
                    (c.num_attention_heads + 2 * c.num_key_value_heads) * c.head_dim,
                    v.pp("attention.query_key_value"),
                )?,
                out: candle_nn::linear_no_bias(
                    c.num_attention_heads * c.head_dim,
                    c.hidden_size,
                    v.pp("attention.dense"),
                )?,
                qnorm: if c.use_qk_norm {
                    Some(Norm::load(
                        c.head_dim,
                        c.rms_norm_eps,
                        v.pp("attention.query_layernorm"),
                    )?)
                } else {
                    None
                },
                knorm: if c.use_qk_norm {
                    Some(Norm::load(
                        c.head_dim,
                        c.rms_norm_eps,
                        v.pp("attention.key_layernorm"),
                    )?)
                } else {
                    None
                },
                mlp: if i < c.first_k_dense_replace {
                    FeedForward::Dense(Mlp::load(c.hidden_size, c.intermediate_size, v.pp("mlp"))?)
                } else {
                    FeedForward::Sparse(Moe::load(&c, v.pp("mlp"))?)
                },
            });
        }
        let norm = Norm::load(c.hidden_size, c.rms_norm_eps, vb.pp("model.norm"))?;
        let head = candle_nn::linear_no_bias(c.hidden_size, c.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            config: c,
            embeddings,
            layers,
            norm,
            head,
            device: device.clone(),
            dtype,
            moe_execution: MoeExecution {
                batched_experts: true,
                compact_decode_experts: false,
                fused_activation: true,
                fused_mix: true,
                device_routing: false,
            },
            fused_attention: true,
            fused_qkv: true,
            prefill_chunk_tokens: 4096,
            attention_chunk_tokens: 1024,
            _lease: lease,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn set_batched_experts(&mut self, enabled: bool) {
        self.moe_execution.batched_experts = enabled;
    }
    pub fn set_compact_decode_experts(&mut self, enabled: bool) {
        self.moe_execution.compact_decode_experts = enabled;
    }
    pub fn set_fused_activation(&mut self, enabled: bool) {
        self.moe_execution.fused_activation = enabled;
    }
    pub fn set_fused_expert_mix(&mut self, enabled: bool) {
        self.moe_execution.fused_mix = enabled;
    }
    pub fn set_device_routing(&mut self, enabled: bool) {
        self.moe_execution.device_routing = enabled;
    }
    pub fn set_fused_norm(&mut self, enabled: bool) {
        self.norm.fused = enabled;
        for layer in &mut self.layers {
            layer.input_norm.fused = enabled;
            layer.post_norm.fused = enabled;
            if let Some(norm) = &mut layer.qnorm {
                norm.fused = enabled;
            }
            if let Some(norm) = &mut layer.knorm {
                norm.fused = enabled;
            }
        }
    }
    pub fn set_fused_attention(&mut self, enabled: bool) {
        self.fused_attention = enabled;
    }
    pub fn set_fused_qkv(&mut self, enabled: bool) {
        self.fused_qkv = enabled;
    }
    pub fn set_prefill_chunk_tokens(&mut self, tokens: usize) -> Result<()> {
        ensure!(
            tokens > 0
                && tokens <= MAX_FORWARD_TOKENS
                && tokens.is_multiple_of(self.config.block_size),
            "prefill chunk size must be a positive whole number of blocks, at most {MAX_FORWARD_TOKENS} tokens"
        );
        self.prefill_chunk_tokens = tokens;
        Ok(())
    }
    pub fn prefill_chunk_tokens(&self) -> usize {
        self.prefill_chunk_tokens
    }

    pub fn set_attention_chunk_tokens(&mut self, tokens: usize) -> Result<()> {
        ensure!(
            tokens > 0 && tokens.is_multiple_of(self.config.block_size),
            "attention chunk size must be a positive whole number of blocks"
        );
        self.attention_chunk_tokens = tokens;
        Ok(())
    }
    pub fn attention_chunk_tokens(&self) -> usize {
        self.attention_chunk_tokens
    }

    fn rope(&self, n: usize, offset: usize) -> Result<(Tensor, Tensor)> {
        let dim = self.config.rotary_dim();
        let mut angles = Vec::with_capacity(n * dim);
        for p in offset..offset + n {
            for i in 0..dim {
                let inv = 1.0f32
                    / (self.config.rope_theta as f32)
                        .powf((2 * (i % (dim / 2))) as f32 / dim as f32);
                angles.push(p as f32 * inv);
            }
        }
        let angles = Tensor::from_vec(angles, (1, n, dim), &self.device)?;
        Ok((
            angles.cos()?.to_dtype(self.dtype)?,
            angles.sin()?.to_dtype(self.dtype)?,
        ))
    }

    fn rotate(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let r = self.config.rotary_dim();
        let rot = x.narrow(2, 0, r)?;
        let half = Tensor::cat(
            &[
                &rot.narrow(2, r / 2, r / 2)?.neg()?,
                &rot.narrow(2, 0, r / 2)?,
            ],
            2,
        )?;
        let y = (rot.broadcast_mul(cos)? + half.broadcast_mul(sin)?)?;
        Ok(if r == self.config.head_dim {
            y
        } else {
            Tensor::cat(&[&y, &x.narrow(2, r, self.config.head_dim - r)?], 2)?
        })
    }

    fn prepare_qkv(
        &self,
        layer: &Layer,
        qkv: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let c = &self.config;
        #[cfg(feature = "cuda")]
        if self.fused_qkv
            && self.device.is_cuda()
            && self.dtype == DType::BF16
            && c.num_attention_heads == 16
            && c.num_key_value_heads == 4
            && c.head_dim == 128
            && c.rotary_dim() == 64
            && let (Some(qnorm), Some(knorm)) = (&layer.qnorm, &layer.knorm)
            && qnorm.fused
            && knorm.fused
        {
            let heads = crate::cuda::prepare_qkv(
                qkv,
                &qnorm.weight,
                &knorm.weight,
                cos,
                sin,
                c.rms_norm_eps,
            )?;
            return Ok((
                heads.narrow(0, 0, 16)?,
                heads.narrow(0, 16, 4)?,
                heads.narrow(0, 20, 4)?,
            ));
        }
        let mut q = qkv.narrow(1, 0, c.num_attention_heads)?.transpose(0, 1)?;
        let mut k = qkv
            .narrow(1, c.num_attention_heads, c.num_key_value_heads)?
            .transpose(0, 1)?;
        let v = qkv
            .narrow(
                1,
                c.num_attention_heads + c.num_key_value_heads,
                c.num_key_value_heads,
            )?
            .transpose(0, 1)?
            .contiguous()?;
        if let Some(norm) = &layer.qnorm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &layer.knorm {
            k = norm.forward(&k)?;
        }
        Ok((
            self.rotate(&q, cos, sin)?.contiguous()?,
            self.rotate(&k, cos, sin)?.contiguous()?,
            v,
        ))
    }

    /// Tile queries independently from the transformer batch. Every tile sees
    /// only its committed prefix and earlier/current blocks within this batch.
    fn attention(
        &self,
        q: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        let c = &self.config;
        let n = q.dim(1)?;
        let groups = c.num_attention_heads / c.num_key_value_heads;
        let budget = MAX_ATTENTION_ELEMENTS / c.num_attention_heads / (offset + n);
        let chunk = n
            .min(self.attention_chunk_tokens)
            .min(budget / c.block_size * c.block_size);
        ensure!(
            chunk >= c.block_size,
            "attention workspace cannot fit one block"
        );
        let mut outputs = Vec::new();
        for start in (0..n).step_by(chunk) {
            let len = chunk.min(n - start);
            let query_offset = offset + start;
            let total = query_offset + len;
            // cuBLAS accepts the cache's head stride directly, without copying
            // the prefix. Keys beyond the end of this tile need not be read.
            let keys = keys.narrow(1, 0, total)?;
            let values = values.narrow(1, 0, total)?;
            let q = q.narrow(1, start, len)?.contiguous()?.reshape((
                c.num_key_value_heads,
                groups * len,
                c.head_dim,
            ))?;
            let raw_scores = q.matmul(&keys.t()?)?;
            let eager_softmax = || -> Result<Tensor> {
                // Preserve the FP32 scale and BF16 rounding before softmax.
                let scores = (raw_scores.to_dtype(DType::F32)? * (c.head_dim as f64).powf(-0.5))?
                    .to_dtype(self.dtype)?
                    .reshape((c.num_key_value_heads, groups, len, total))?;
                let scores = if len > c.block_size {
                    let mask: Vec<f32> = (0..len)
                        .flat_map(|q| {
                            (0..total).map(move |k| {
                                if k / c.block_size <= (query_offset + q) / c.block_size {
                                    0.
                                } else {
                                    f32::NEG_INFINITY
                                }
                            })
                        })
                        .collect();
                    let mask = Tensor::from_vec(mask, (1, 1, len, total), &self.device)?
                        .to_dtype(self.dtype)?;
                    scores.broadcast_add(&mask)?
                } else {
                    scores
                };
                Ok(
                    candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
                        .to_dtype(self.dtype)?
                        .reshape((c.num_key_value_heads, groups * len, total))?,
                )
            };
            #[cfg(feature = "cuda")]
            let probs = if self.fused_attention
                && self.device.is_cuda()
                && self.dtype == DType::BF16
                && total <= 8192
            {
                crate::cuda::block_softmax(
                    &raw_scores,
                    len,
                    query_offset,
                    c.block_size,
                    (c.head_dim as f64).powf(-0.5) as f32,
                )?
            } else {
                eager_softmax()?
            };
            #[cfg(not(feature = "cuda"))]
            let probs = eager_softmax()?;
            outputs.push(
                probs
                    .matmul(&values)?
                    .reshape((c.num_attention_heads, len, c.head_dim))?
                    .transpose(0, 1)?
                    .contiguous()?
                    .reshape((len, c.num_attention_heads * c.head_dim))?,
            );
        }
        Ok(Tensor::cat(&outputs, 0)?)
    }

    /// Evaluates aligned blocks at the committed offset, replacing uncommitted K/V.
    /// `logits=false` avoids the large vocabulary projection during prefill/commit.
    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut Cache,
        logits: bool,
        mut trace: Option<&mut Trace>,
    ) -> Result<Option<Tensor>> {
        let c = &self.config;
        let n = tokens.len();
        ensure!(
            n > 0 && n.is_multiple_of(c.block_size) && cache.len.is_multiple_of(c.block_size),
            "forward must contain complete aligned blocks"
        );
        ensure!(
            cache.layers.len() == c.num_hidden_layers && cache.len + n <= cache.capacity,
            "cache capacity exceeded or incompatible cache"
        );
        ensure!(
            tokens.iter().all(|&t| (t as usize) < c.vocab_size),
            "token outside vocabulary"
        );
        ensure!(
            n <= MAX_FORWARD_TOKENS,
            "forward workspace limit exceeded; process at most {MAX_FORWARD_TOKENS} tokens per batch"
        );
        cache.staged_tokens = None;
        let offset = cache.len;
        let mut x = self
            .embeddings
            .index_select(&Tensor::from_vec(tokens.to_vec(), n, &self.device)?, 0)?;
        record(&mut trace, "embeddings".into(), &x);
        if !matches!(&cache.rope, Some((position, count, _, _)) if *position==offset && *count==n) {
            let (cos, sin) = self.rope(n, offset)?;
            cache.rope = Some((offset, n, cos, sin));
        }
        let (_, _, cos, sin) = cache.rope.as_ref().unwrap();
        let (cos, sin) = (cos.clone(), sin.clone());
        for (i, layer) in self.layers.iter().enumerate() {
            let h = layer.input_norm.forward(&x)?;
            let qkv = layer.qkv.forward(&h)?.reshape((
                n,
                c.num_attention_heads + 2 * c.num_key_value_heads,
                c.head_dim,
            ))?;
            let (q, k, v) = self.prepare_qkv(layer, &qkv, &cos, &sin)?;
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
            let h = self.attention(&q, keys, values, offset)?;
            x = (x + layer.out.forward(&h)?)?;
            record(&mut trace, format!("layers.{i}.attention"), &x);
            let h = layer.post_norm.forward(&x)?;
            let y = match &layer.mlp {
                FeedForward::Dense(mlp) => mlp.forward(&h, self.moe_execution.fused_activation)?,
                FeedForward::Sparse(moe) => moe.forward(
                    &h,
                    c,
                    &mut trace,
                    &format!("layers.{i}"),
                    &self.moe_execution,
                )?,
            };
            x = (x + y).with_context(|| format!("layer {i}"))?;
            record(&mut trace, format!("layers.{i}.hidden"), &x);
        }
        let h = self.norm.forward(&x)?;
        record(&mut trace, "normalized".into(), &h);
        let out = if logits {
            Some(self.head.forward(&h)?.to_dtype(DType::F32)?)
        } else {
            None
        };
        cache.staged_tokens = Some(tokens.to_vec());
        cache.forwards += 1;
        cache.processed_tokens += n;
        Ok(out)
    }
}
