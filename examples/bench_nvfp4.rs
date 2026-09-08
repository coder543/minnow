#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use minnow::{
        container::Encoding,
        quant::{Weights, nvfp4},
    };
    let dev = Device::new_cuda(0)?;
    let pool = minnow::cuda::workspace::WorkspaceCache::new(&dev).unwrap();
    pool.set(1024 * 1024 * 1024)?;
    let mut reports = Vec::new();
    let experts = 48;
    for (out, input) in [(512, 2048), (2048, 512), (1024, 4096), (4096, 1024)] {
        if std::env::var("MINNOW_BENCH_INPUT")
            .ok()
            .is_some_and(|v| v != input.to_string())
        {
            continue;
        }
        let mut codes: Vec<u8> = Vec::new();
        let mut scales: Vec<u8> = Vec::new();
        let mut globals = Vec::new();
        let values: Vec<f32> = (0..out * input)
            .map(|i| ((i * 17 % 257) as f32 - 128.) / 1000.)
            .collect();
        let (c, s, g) = nvfp4::encode(&values)?;
        let (c, s) = nvfp4::pack(&c, &s, input, false)?;
        for _ in 0..experts {
            codes.extend(&c);
            scales.extend(&s);
            globals.push(g);
        }
        let w = Weights {
            codes: Tensor::from_vec(codes, experts * out * input / 2, &dev)?,
            scales: Tensor::from_vec(scales, experts * out * input / 16, &dev)?,
            global_scales: Some(Tensor::from_vec(globals, experts, &dev)?),
            encoding: Encoding::Nvfp4,
            group_size: 16,
            shape: (experts, out, input),
        };
        let values: Vec<f32> = values
            .into_iter()
            .cycle()
            .take(experts * out * input)
            .collect();
        let bf16 =
            Tensor::from_slice(&values, (experts, out, input), &dev)?.to_dtype(DType::BF16)?;
        drop(values);
        for n in [6, 32, 128] {
            if std::env::var("MINNOW_BENCH_ROWS")
                .ok()
                .is_some_and(|v| v != n.to_string())
            {
                continue;
            }
            let x = Tensor::from_vec(
                (0..experts * n * input)
                    .map(|i| ((i * 13 % 101) as f32 - 50.) / 59.)
                    .collect::<Vec<_>>(),
                (experts * n, input),
                &dev,
            )?
            .to_dtype(DType::BF16)?;
            let segments: Vec<_> = (0..experts).map(|e| (e, n)).collect();
            for mode in ["nvfp4", "bf16"] {
                if std::env::var("MINNOW_BENCH_MODE")
                    .ok()
                    .is_some_and(|v| v != mode)
                {
                    continue;
                }
                let run = || -> anyhow::Result<()> {
                    match mode {
                        "nvfp4" => {
                            let _ = w.grouped(&x, &segments)?;
                        }
                        _ => {
                            let _ = minnow::cuda::grouped_expert_gemm(&x, &bf16, &segments)?;
                        }
                    }
                    dev.synchronize()?;
                    Ok(())
                };
                pool.refresh()?;
                for _ in 0..5 {
                    run()?;
                }
                let start = std::time::Instant::now();
                for _ in 0..30 {
                    run()?;
                }
                reports.push(serde_json::json!({"encoding":mode,"out":out,"input":input,"experts":experts,"rows_per_expert":n,"milliseconds":start.elapsed().as_secs_f64()*1000./30.}));
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("requires CUDA");
}
