//! Locate the first batch-boundary-dependent divergence, without slot copying.
use anyhow::{Result, ensure};
use candle_core::{DType, Device};
use minnow::{
    decode::prefill,
    model::{Cache, Model, Trace},
};
use serde_json::json;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() >= 3,
        "usage: trace_prefill_boundaries CHECKPOINT IDS.json SPLIT [f32] [strict]"
    );
    let dtype = if args.iter().any(|a| a == "f32") {
        DType::F32
    } else {
        DType::BF16
    };
    let device = Device::new_cuda(0)?;
    #[cfg(feature = "cuda")]
    if args.iter().any(|a| a == "strict")
        && let Device::Cuda(dev) = &device
    {
        use candle_core::cuda_backend::cudarc::cublas::sys;
        // SAFETY: this diagnostic owns the idle device/handle for its lifetime.
        unsafe {
            sys::cublasSetMathMode(
                *dev.cublas_handle().handle(),
                sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
            .result()?;
        }
    }
    let ids: Vec<u32> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let split: usize = args[2].parse()?;
    let model = Model::load(Path::new(&args[0]), dtype, &device)?;
    let chunk = model.prefill_chunk_tokens();
    let base = split / chunk * chunk;
    let end = (base + chunk).min(ids.len() / 32 * 32);
    ensure!(
        split > base && split < end && split.is_multiple_of(32),
        "split must be inside a prefill batch"
    );
    let run = |end: usize| -> Result<Trace> {
        let mut cache = Cache::new(&model.config, ids.len().div_ceil(32) * 32)?;
        prefill(&model, &ids[..base], &mut cache, || false)?;
        let mut trace = Trace::new();
        model.forward(&ids[base..end], &mut cache, false, Some(&mut trace))?;
        device.synchronize()?;
        Ok(trace)
    };
    let mut cold = run(end)?;
    let mut split_trace = run(split)?;
    let mut keys = vec!["embeddings".to_string()];
    for layer in 0..model.config.num_hidden_layers {
        for kind in [
            "attention",
            "router_input",
            "router_logits",
            "router_ids",
            "hidden",
        ] {
            let key = format!("layers.{layer}.{kind}");
            if cold.contains_key(&key) {
                keys.push(key);
            }
        }
    }
    keys.push("normalized".to_string());
    for key in keys {
        let a = cold.remove(&key).unwrap().narrow(0, 0, split - base)?;
        let b = split_trace.remove(&key).unwrap();
        if key.ends_with("router_ids") {
            let a = a.to_vec2::<u32>()?;
            let b = b.to_vec2::<u32>()?;
            let order = a.iter().zip(&b).filter(|(a, b)| a != b).count();
            let sets = a
                .into_iter()
                .zip(b)
                .filter(|(a, b)| {
                    let mut a = a.clone();
                    let mut b = b.clone();
                    a.sort_unstable();
                    b.sort_unstable();
                    a != b
                })
                .count();
            println!(
                "{}",
                json!({"name":key,"rows_with_different_order":order,"rows_with_different_expert_sets":sets,"rows":split-base})
            );
        } else {
            let difference = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?;
            println!(
                "{}",
                json!({"name":key,"max_abs_error":difference.abs()?.max_all()?.to_scalar::<f32>()?,
                "rms_error":difference.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?})
            );
        }
    }
    Ok(())
}
