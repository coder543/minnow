//! Bounded synthetic fixture for profiling the BF16 mini vocabulary projection.
#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use candle_core::{Device, Tensor};
    use half::bf16;
    let dev = Device::new_cuda(0)?;
    let (rows, hidden, vocab) = (32, 2048, 157184);
    let values = |n: usize| {
        (0..n)
            .map(|i| bf16::from_f32(((i * 17 % 257) as f32 - 128.) / 1000.))
            .collect::<Vec<_>>()
    };
    let w = Tensor::from_vec(values(vocab * hidden), (vocab, hidden), &dev)?;
    let x = Tensor::from_vec(values(rows * hidden), (rows, hidden), &dev)?;
    for _ in 0..6 {
        let _ = x.matmul(&w.t()?)?;
        dev.synchronize()?;
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("requires CUDA");
}
