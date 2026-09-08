//! Compare fixed expert work across BF16, INT8, and native NVFP4.
//! This is a bounded synthetic projection fixture, not a model/quality benchmark.
#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use minnow::{
        container::Encoding,
        quant::{Weights, encode_matrix, nvfp4},
    };
    let dev = Device::new_cuda(0)?;
    let pool = minnow::cuda::workspace::WorkspaceCache::new(&dev).unwrap();
    pool.set(1024 * 1024 * 1024)?;
    let filter = |key: &str, value: &str| std::env::var(key).is_ok_and(|v| v != value);
    let mut reports = Vec::new();
    let experts = 48;
    for (out, input) in [(512, 2048), (2048, 512), (1024, 4096), (4096, 1024)] {
        if filter("MINNOW_BENCH_INPUT", &input.to_string()) {
            continue;
        }
        let values: Vec<f32> = (0..out * input)
            .map(|i| ((i * 17 % 257) as f32 - 128.) / 1000.)
            .collect();
        for mode in ["bf16", "int8", "nvfp4"] {
            if filter("MINNOW_BENCH_MODE", mode) {
                continue;
            }
            let (float, weight) = match mode {
                "bf16" => {
                    let values: Vec<half::bf16> = values
                        .iter()
                        .copied()
                        .map(half::bf16::from_f32)
                        .cycle()
                        .take(experts * out * input)
                        .collect();
                    (
                        Some(Tensor::from_vec(values, (experts, out, input), &dev)?),
                        None,
                    )
                }
                _ => {
                    let native = mode == "nvfp4";
                    let encoding = if native {
                        Encoding::Nvfp4
                    } else {
                        Encoding::I8Mma
                    };
                    let (codes, scales, global) = if native {
                        let (c, s, g) = nvfp4::encode(&values)?;
                        let (c, s) = nvfp4::pack(&c, &s, input, false)?;
                        (c, s, Some(g))
                    } else {
                        let (c, s) = encode_matrix(&values, encoding, 128, input)?;
                        (c, s, None)
                    };
                    let c: Vec<u8> = codes
                        .iter()
                        .copied()
                        .cycle()
                        .take(experts * codes.len())
                        .collect();
                    let scales = if native {
                        let s: Vec<u8> = scales
                            .iter()
                            .copied()
                            .cycle()
                            .take(experts * scales.len())
                            .collect();
                        Tensor::from_vec(s, experts * scales.len(), &dev)?
                    } else {
                        let s: Vec<half::f16> = scales
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|s| half::f16::from_bits(u16::from_le_bytes(*s)))
                            .cycle()
                            .take(experts * scales.len() / 2)
                            .collect();
                        Tensor::from_vec(s, experts * scales.len() / 2, &dev)?
                    };
                    (
                        None,
                        Some(Weights {
                            codes: Tensor::from_vec(c, experts * codes.len(), &dev)?,
                            scales,
                            global_scales: global
                                .map(|g| Tensor::from_vec(vec![g; experts], experts, &dev))
                                .transpose()?,
                            encoding,
                            group_size: if native { 16 } else { 128 },
                            shape: (experts, out, input),
                        }),
                    )
                }
            };
            for n in [6, 32, 128] {
                if filter("MINNOW_BENCH_ROWS", &n.to_string()) {
                    continue;
                }
                let x = Tensor::from_vec(
                    (0..experts * n * input)
                        .map(|i| half::bf16::from_f32(((i * 13 % 101) as f32 - 50.) / 59.))
                        .collect::<Vec<_>>(),
                    (experts * n, input),
                    &dev,
                )?;
                let segments: Vec<_> = (0..experts).map(|e| (e, n)).collect();
                let run = || -> anyhow::Result<()> {
                    let _ = if let Some(w) = &weight {
                        w.grouped(&x, &segments)?
                    } else {
                        minnow::cuda::grouped_expert_gemm(&x, float.as_ref().unwrap(), &segments)?
                    };
                    dev.synchronize()?;
                    Ok(())
                };
                pool.refresh()?;
                for _ in 0..5 {
                    run()?;
                }
                let mut times = Vec::new();
                for _ in 0..3 {
                    let start = std::time::Instant::now();
                    for _ in 0..20 {
                        run()?;
                    }
                    times.push(start.elapsed().as_secs_f64() * 1000. / 20.);
                }
                let ms = times.iter().sum::<f64>() / times.len() as f64;
                reports.push(serde_json::json!({"encoding":mode,"out":out,"input":input,"experts":experts,"rows_per_expert":n,"milliseconds":ms,"repetitions_ms":times,"activation_dtype":format!("{:?}",DType::BF16)}));
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("requires --features cuda");
}
