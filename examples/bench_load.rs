//! Measure checkpoint loading separately from CUDA initialization and inference.
#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use candle_core::{DType, Device};
    use minnow::model::Model;
    use std::{path::Path, time::Instant};
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() == 2, "usage: bench_load CHECKPOINT");
    let device = Device::new_cuda(0)?;
    let start = Instant::now();
    let model = Model::load(Path::new(&args[1]), DType::BF16, &device)?;
    device.synchronize()?;
    println!(
        "{}",
        serde_json::json!({"checkpoint":args[1],"load_seconds":start.elapsed().as_secs_f64()})
    );
    drop(model);
    device.synchronize()?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("bench_load requires --features cuda");
}
