//! Compare one-shot and resumed BF16 prefill with identical K/V capacity.
//! Run under memory_guard.py, sequentially with any full-model server stopped.
use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use minnow::{
    decode::prefill,
    model::{Cache, Model},
};
use serde_json::json;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(
        args.len() == 3 || (args.len() == 4 && args[3] == "f32"),
        "usage: check_prefix_numerics CHECKPOINT IDS.json REUSE_TOKENS [f32]"
    );
    let ids: Vec<u32> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let split: usize = args[2].parse()?;
    let device = Device::new_cuda(0)?;
    let dtype = if args.len() == 4 {
        DType::F32
    } else {
        DType::BF16
    };
    let model = Model::load(&PathBuf::from(&args[0]), dtype, &device)?;
    let b = model.config.block_size;
    let n = ids.len() / b * b;
    ensure!(
        split > 0 && split < n && split.is_multiple_of(b),
        "invalid reusable prefix"
    );
    let mut query = vec![156895; b];
    query[..ids.len() - n].copy_from_slice(&ids[n..]);
    let run = |split: Option<usize>| -> Result<Tensor> {
        let mut cache = Cache::new(&model.config, n + b)?;
        if let Some(split) = split {
            prefill(&model, &ids[..split], &mut cache, || false)?;
        }
        prefill(&model, &ids[..n], &mut cache, || false)?;
        let logits = model.forward(&query, &mut cache, true, None)?.unwrap();
        device.synchronize()?;
        Ok(logits)
    };
    let cold = run(None)?;
    let resumed = run(Some(split))?;
    let difference = (&cold - &resumed)?;
    let error = difference.abs()?.max_all()?.to_scalar::<f32>()?;
    let rms = difference.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    let cold_ids = cold.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?;
    let resumed_ids = resumed.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?;
    println!(
        "{}",
        json!({"dtype":format!("{dtype:?}"),"prompt_tokens":ids.len(),"split":split,"max_abs_error":error,
        "rms_error":rms,"top1_matching_rows":cold_ids.iter().zip(&resumed_ids).filter(|(a,b)|a==b).count(),
        "rows":b,"cache_capacity":n+b,"description":"same model and K/V capacity; only prefill batch boundaries differ"})
    );
    Ok(())
}
