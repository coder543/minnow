use candle_core::{DType, Device, Tensor};
use minnow::model::{Cache, Model, Trace};
use std::path::Path;

fn assert_close(a: &Tensor, b: &Tensor, tolerance: f32) {
    assert_eq!(a.dims(), b.dims());
    let a = a
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let b = b
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let max = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        a.iter().chain(&b).all(|v| v.is_finite()) && max <= tolerance,
        "maximum absolute error {max} > {tolerance}"
    );
}

#[test]
fn checkpoint_reference_matches_every_layer_and_routing() {
    let path = Path::new("tests/fixtures/tiny");
    let model = Model::load(path, DType::F32, &Device::Cpu).unwrap();
    let tokens: Vec<u32> =
        serde_json::from_slice(&std::fs::read(path.join("input.json")).unwrap()).unwrap();
    let expected =
        candle_core::safetensors::load(path.join("reference.safetensors"), &Device::Cpu).unwrap();
    let mut cache = Cache::new(&model.config, tokens.len()).unwrap();
    let mut trace = Trace::new();
    let logits = model
        .forward(&tokens, &mut cache, true, Some(&mut trace))
        .unwrap()
        .unwrap();
    trace.insert("logits".into(), logits);
    for (name, target) in &expected {
        eprintln!("comparing {name}");
        assert_close(&trace[name], target, 1e-4);
    }
}

#[test]
fn refinement_and_final_commit_match_recomputed_prefix() {
    let path = Path::new("tests/fixtures/tiny");
    let model = Model::load(path, DType::F32, &Device::Cpu).unwrap();
    let mut tokens: Vec<u32> =
        serde_json::from_slice(&std::fs::read(path.join("input.json")).unwrap()).unwrap();
    let b = model.config.block_size;
    let mut cache = Cache::new(&model.config, tokens.len()).unwrap();
    minnow::decode::prefill(&model, &tokens[..b], &mut cache, || false).unwrap();
    let old = model
        .forward(&tokens[b..2 * b], &mut cache, true, None)
        .unwrap()
        .unwrap();
    // Simulate edits after a forward. A stale scratch cache cannot be committed.
    tokens[b + 3] = 111;
    tokens[b + 8] = 222;
    assert!(cache.commit(&tokens[b..2 * b]).is_err());
    assert_eq!(cache.len(), b);
    let refined = model
        .forward(&tokens[b..2 * b], &mut cache, true, None)
        .unwrap()
        .unwrap();
    assert!(
        (old - refined.clone())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            > 1e-4
    );
    let mut full = Cache::new(&model.config, tokens.len()).unwrap();
    let target = model
        .forward(&tokens[..2 * b], &mut full, true, None)
        .unwrap()
        .unwrap();
    assert_close(&refined, &target.narrow(0, b, b).unwrap(), 1e-4);
    cache.commit(&tokens[b..2 * b]).unwrap();
    let last = model
        .forward(&tokens[2 * b..], &mut cache, true, None)
        .unwrap()
        .unwrap();
    let target = model
        .forward(&tokens, &mut full, true, None)
        .unwrap()
        .unwrap();
    assert_close(&last, &target.narrow(0, 2 * b, b).unwrap(), 1e-4);
    assert!(cache.commit(&tokens[..b - 1]).is_err());
}

#[test]
fn block_commit_reuses_current_kv_and_refreshes_edited_tokens() {
    let path = Path::new("tests/fixtures/tiny");
    let model = Model::load(path, DType::F32, &Device::Cpu).unwrap();
    let seed: Vec<u32> =
        serde_json::from_slice(&std::fs::read(path.join("input.json")).unwrap()).unwrap();
    let b = model.config.block_size;
    for edited in [false, true] {
        let mut tokens = seed.clone();
        let mut cache = Cache::new(&model.config, tokens.len()).unwrap();
        minnow::decode::prefill(&model, &tokens[..b], &mut cache, || false).unwrap();
        model
            .forward(&tokens[b..2 * b], &mut cache, true, None)
            .unwrap();
        if edited {
            tokens[b + 3] = 111;
            tokens[b + 8] = 222;
            assert!(cache.commit(&tokens[b..2 * b]).is_err());
        }
        let forwards = cache.forwards;
        let refreshed =
            minnow::decode::commit_block(&model, &tokens[b..2 * b], &mut cache).unwrap();
        assert_eq!(refreshed, edited);
        assert_eq!(cache.forwards, forwards + usize::from(edited));
        assert_eq!(cache.len(), 2 * b);
        // A committed block cannot be committed twice without another forward.
        assert!(cache.commit(&tokens[b..2 * b]).is_err());
        let actual = model
            .forward(&tokens[2 * b..], &mut cache, true, None)
            .unwrap()
            .unwrap();
        let mut full = Cache::new(&model.config, tokens.len()).unwrap();
        let expected = model
            .forward(&tokens, &mut full, true, None)
            .unwrap()
            .unwrap();
        assert_close(&actual, &expected.narrow(0, 2 * b, b).unwrap(), 1e-4);
    }
}

#[test]
fn prefill_batch_boundaries_preserve_future_block_logits() {
    let path = Path::new("tests/fixtures/tiny");
    let mut model = Model::load(path, DType::F32, &Device::Cpu).unwrap();
    let seed: Vec<u32> =
        serde_json::from_slice(&std::fs::read(path.join("input.json")).unwrap()).unwrap();
    let tokens: Vec<u32> = seed.iter().copied().cycle().take(256).collect();
    let mut full = Cache::new(&model.config, tokens.len()).unwrap();
    let expected = model
        .forward(&tokens, &mut full, true, None)
        .unwrap()
        .unwrap()
        .narrow(0, 224, 32)
        .unwrap();
    // Include a partial final chunk and the serving default, larger than this fixture.
    for (chunk_size, attention_chunk) in [(32, 32), (96, 32), (128, 96), (2048, 32), (2048, 1024)] {
        model.set_prefill_chunk_tokens(chunk_size).unwrap();
        model.set_attention_chunk_tokens(attention_chunk).unwrap();
        let mut cache = Cache::new(&model.config, tokens.len()).unwrap();
        minnow::decode::prefill(&model, &tokens[..224], &mut cache, || false).unwrap();
        assert_eq!(cache.len(), 224);
        let actual = model
            .forward(&tokens[224..], &mut cache, true, None)
            .unwrap()
            .unwrap();
        assert_close(&actual, &expected, 1e-4);
    }
}

#[test]
fn observed_generation_preserves_outputs_and_reports_only_final_blocks() {
    use minnow::decode::{Options, Progress, SpecialTokens, generate, generate_observed};
    let model = Model::load(Path::new("tests/fixtures/tiny"), DType::F32, &Device::Cpu).unwrap();
    let prompt = vec![1; 35];
    let options = Options {
        max_tokens: Some(64),
        steps: 1,
        max_post_steps: 0,
        ..Options::default()
    };
    let special = SpecialTokens {
        mask: 258,
        delete: 256,
        split: 257,
        eos: 255,
    };
    let expected = generate(&model, &prompt, &options, special, || false).unwrap();
    let mut final_tokens = Vec::new();
    let mut refinements = 0;
    let mut prefill = Vec::new();
    let actual = generate_observed(
        &model,
        &prompt,
        &options,
        special,
        || false,
        |event| {
            match event {
                Progress::Prefill {
                    processed, total, ..
                } => {
                    assert_eq!(total, 32);
                    prefill.push(processed);
                }
                Progress::Refinement { stats, batch } => {
                    refinements += 1;
                    assert_eq!(stats.evaluated_tokens, refinements * 32);
                    assert_eq!(batch.evaluated_tokens, batch.refinement_steps * 32);
                }
                Progress::Block { token_ids, .. } => {
                    assert!(token_ids.starts_with(&final_tokens));
                    assert!(!token_ids.contains(&special.mask));
                    final_tokens = token_ids.to_vec();
                }
            }
            Ok(true)
        },
    )
    .unwrap();
    assert_eq!(prefill, [0, 32]);
    assert_eq!(actual.token_ids, expected.token_ids);
    assert_eq!(final_tokens, expected.token_ids);
    assert_eq!(actual.stats.total_forwards, expected.stats.total_forwards);
    assert_eq!(
        actual.stats.evaluated_tokens,
        actual
            .batches
            .iter()
            .map(|b| b.evaluated_tokens)
            .sum::<usize>()
    );
    assert_eq!(
        actual.stats.completion_tokens,
        actual
            .batches
            .iter()
            .map(|b| b.completion_tokens)
            .sum::<usize>()
    );
    let stopped = generate_observed(
        &model,
        &prompt,
        &options,
        special,
        || false,
        |event| Ok(!matches!(event, Progress::Block { .. })),
    )
    .unwrap();
    assert_eq!(stopped.finish_reason, "stop");
    assert_eq!(stopped.stats.blocks, 1);
    assert!(actual.token_ids.starts_with(&stopped.token_ids));
}

#[test]
fn huge_full_prefix_is_rejected_before_attention_allocation() {
    let mut model =
        Model::load(Path::new("tests/fixtures/tiny"), DType::F32, &Device::Cpu).unwrap();
    model.config.max_position_embeddings = 131072;
    let mut cache = Cache::new(&model.config, 131072).unwrap();
    let error = model
        .forward(&vec![1; 131072], &mut cache, false, None)
        .unwrap_err();
    assert!(error.to_string().contains("workspace limit"));
    assert_eq!(cache.forwards, 0);
    assert!(cache.is_empty());
}
