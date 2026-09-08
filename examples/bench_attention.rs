#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    let dev = Device::new_cuda(0)?;
    let pool = minnow::cuda::workspace::WorkspaceCache::new(&dev).unwrap();
    pool.set(1024 * 1024 * 1024)?;
    let mut reports = Vec::new();
    for (offset, n) in [(0, 4096), (0, 8192), (4096, 32)] {
        let total = offset + n;
        let make = |h, n, seed| -> candle_core::Result<Tensor> {
            Tensor::from_vec(
                (0..h * n * 128)
                    .map(|i| ((i * seed % 1009) as f32 - 504.) / 800.)
                    .collect::<Vec<_>>(),
                (h, n, 128),
                &dev,
            )?
            .to_dtype(DType::BF16)
        };
        let q = make(16, n, 17)?;
        let k = make(4, total, 31)?;
        let v = make(4, total, 47)?;
        for mode in ["materialized", "flash"] {
            let run = || -> anyhow::Result<()> {
                if mode == "flash" {
                    let _ = minnow::cuda::flash::attention(&q, &k, &v, offset)?;
                } else {
                    let mut outputs = Vec::new();
                    for start in (0..n).step_by(1024) {
                        let len = (n - start).min(1024);
                        let visible = offset + start + len;
                        let q =
                            q.narrow(1, start, len)?
                                .contiguous()?
                                .reshape((4, 4 * len, 128))?;
                        let scores = q.matmul(&k.narrow(1, 0, visible)?.t()?)?;
                        let p = minnow::cuda::block_softmax(
                            &scores,
                            len,
                            offset + start,
                            32,
                            128f32.sqrt().recip(),
                        )?;
                        outputs.push(
                            p.matmul(&v.narrow(1, 0, visible)?)?
                                .reshape((16, len, 128))?
                                .transpose(0, 1)?
                                .contiguous()?
                                .reshape((len, 2048))?,
                        );
                    }
                    let _ = Tensor::cat(&outputs, 0)?;
                }
                dev.synchronize()?;
                Ok(())
            };
            pool.refresh()?;
            for _ in 0..3 {
                run()?;
            }
            let start = std::time::Instant::now();
            for _ in 0..10 {
                run()?;
            }
            reports.push(serde_json::json!({"offset":offset,"queries":n,"mode":mode,"milliseconds":start.elapsed().as_secs_f64()*1000./10.}));
        }
    }
    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("requires CUDA");
}
