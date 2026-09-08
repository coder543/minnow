//! Compare attention backends with one resident checkpoint and fixed inputs.
#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use candle_core::{DType, Device};
    use minnow::{
        decode::prefill,
        model::{Cache, Model, Trace},
    };
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 2 || (args.len() == 3 && args[2] == "--trace"),
        "usage: compare_attention CHECKPOINT [--trace]"
    );
    let dev = Device::new_cuda(0)?;
    let mut model = Model::load(std::path::Path::new(&args[1]), DType::BF16, &dev)?;
    model.set_device_routing(true);
    let seed: Vec<u32> =
        serde_json::from_slice(&std::fs::read("tests/prefill_benchmark_ids.json")?)?;
    let mut reports = Vec::new();
    for n in [0, 128, 512, 2048, 8192] {
        let prefix: Vec<_> = seed.iter().copied().cycle().take(n).collect();
        let current: Vec<_> = seed
            .iter()
            .copied()
            .cycle()
            .skip(n)
            .take(32)
            .enumerate()
            .map(|(i, t)| if i % 2 == 0 { 156895 } else { t })
            .collect();
        let mut logits = Vec::new();
        let mut seconds = Vec::new();
        let mut traces = Vec::new();
        for flash in [false, true] {
            model.set_flash_attention(flash);
            let mut cache = Cache::new(&model.config, n + 32)?;
            prefill(&model, &prefix, &mut cache, || false)?;
            let start = std::time::Instant::now();
            let mut trace = Trace::new();
            let tracing = (args.len() == 3 && n == 0).then_some(&mut trace);
            let out = model.forward(&current, &mut cache, true, tracing)?.unwrap();
            traces.push(trace);
            dev.synchronize()?;
            seconds.push(start.elapsed().as_secs_f64());
            logits.push(out.to_dtype(DType::F32)?.to_vec2::<f32>()?);
        }
        let mut layer_errors = Vec::new();
        if !traces[0].is_empty() {
            for layer in 0..model.config.num_hidden_layers {
                for suffix in [
                    "query",
                    "key",
                    "value",
                    "attention_output",
                    "attention",
                    "hidden",
                ] {
                    let key = format!("layers.{layer}.{suffix}");
                    let a = traces[0][&key]
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let b = traces[1][&key]
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let max = a
                        .iter()
                        .zip(&b)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let rms = (a
                        .iter()
                        .zip(&b)
                        .map(|(a, b)| (a - b).powi(2) as f64)
                        .sum::<f64>()
                        / a.len() as f64)
                        .sqrt();
                    layer_errors.push(serde_json::json!({"name":key,"max":max,"rms":rms}));
                }
            }
        }
        let (mut max, mut sq, mut count, mut agree, mut kl) = (0f64, 0f64, 0usize, 0usize, 0f64);
        for (a, b) in logits[0].iter().zip(&logits[1]) {
            let top = |x: &Vec<f32>| {
                x.iter()
                    .enumerate()
                    .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then(ib.cmp(ia)))
                    .unwrap()
                    .0
            };
            agree += usize::from(top(a) == top(b));
            let lse = |x: &Vec<f32>| {
                let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                m + x.iter().map(|v| (*v as f64 - m).exp()).sum::<f64>().ln()
            };
            let (la, lb) = (lse(a), lse(b));
            for (&a, &b) in a.iter().zip(b) {
                let d = (a - b) as f64;
                max = max.max(d.abs());
                sq += d * d;
                count += 1;
                kl += (a as f64 - la).exp() * (d + lb - la);
            }
        }
        reports.push(serde_json::json!({"prefix_tokens":n,"positions":32,"top1_agreement":agree,"maximum_logit_error":max,"rms_logit_error":(sq/count as f64).sqrt(),"mean_kl_baseline_to_flash":kl/32.,"forward_seconds":seconds,"layer_errors":layer_errors}));
    }
    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("requires CUDA");
}
