#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use half::f16;
    use minnow::{
        container::Encoding,
        quant::{Weights, encode_matrix},
    };
    let dev = Device::new_cuda(0)?;
    if let Ok(value) = std::env::var("MINNOW_BENCH_POOL_MIB") {
        use candle_core::cuda_backend::cudarc::driver::sys;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let mut threshold = value.parse::<u64>()? * 1024 * 1024;
        let mut pool = std::ptr::null_mut();
        // SAFETY: the benchmark owns this context and supplies the documented
        // u64 threshold type to the default pool's attribute API.
        unsafe {
            sys::cuDeviceGetDefaultMemPool(&mut pool, cuda.cuda_stream().context().cu_device())
                .result()?;
            sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut threshold as *mut u64).cast(),
            )
            .result()?;
        }
    }
    let mut reports = Vec::new();
    for (out, input) in [(512, 2048), (2048, 512), (1024, 4096), (4096, 1024)] {
        let experts = 48;
        let values: Vec<f32> = (0..experts * out * input)
            .map(|i| ((i * 17 % 257) as f32 - 128.) / 1000.)
            .collect();
        let baseline =
            Tensor::from_slice(&values, (experts, out, input), &dev)?.to_dtype(DType::BF16)?;
        for encoding in [Encoding::I8Sym, Encoding::I8Mma] {
            let group_size = 128;
            let (codes, scales) = encode_matrix(&values, encoding, group_size, input)?;
            let scales: Vec<f16> = scales
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16::from_bits(u16::from_le_bytes(*b)))
                .collect();
            let weight = Weights {
                int8_activations: false,
                global_scales: None,
                codes: Tensor::from_vec(codes.clone(), codes.len(), &dev)?,
                scales: Tensor::from_vec(scales.clone(), scales.len(), &dev)?,
                encoding,
                group_size,
                shape: (experts, out, input),
            };
            for n in [6, 32, 128] {
                let x = Tensor::ones((experts * n, input), DType::BF16, &dev)?;
                let segments: Vec<_> = (0..experts).map(|e| (e, n)).collect();
                for quantized in [false, true] {
                    let run = || -> anyhow::Result<()> {
                        if quantized {
                            let _ = weight.grouped(&x, &segments)?;
                        } else {
                            let _ = minnow::cuda::grouped_expert_gemm(&x, &baseline, &segments)?;
                        }
                        dev.synchronize()?;
                        Ok(())
                    };
                    for _ in 0..3 {
                        run()?;
                    }
                    let start = std::time::Instant::now();
                    for _ in 0..10 {
                        run()?;
                    }
                    let ms = start.elapsed().as_secs_f64() * 100.;
                    reports.push(serde_json::json!({"out":out,"input":input,"experts":experts,"rows_per_expert":n,"encoding":if quantized {format!("{encoding:?}")} else {"BF16".into()},"milliseconds":ms}));
                }
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}
