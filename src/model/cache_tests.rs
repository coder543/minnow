use super::*;

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA"]
fn cuda_model_drops_on_a_thread_that_has_not_submitted_work() -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    let device = Device::new_cuda(0)?;
    let used = || -> Result<u64> {
        let Device::Cuda(cuda) = &device else {
            unreachable!()
        };
        let stream = cuda.cuda_stream();
        stream.context().bind_to_thread()?;
        stream.synchronize()?;
        let mut pool = std::ptr::null_mut();
        let mut bytes = 0u64;
        // SAFETY: current context and correctly typed attribute outputs.
        unsafe {
            sys::cuDeviceGetMemPool(&mut pool, stream.context().cu_device()).result()?;
            sys::cuMemPoolGetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                (&mut bytes as *mut u64).cast(),
            )
            .result()?;
        }
        Ok(bytes)
    };
    let baseline = used()?;
    let model = Model::load(Path::new("tests/fixtures/tiny"), DType::BF16, &device)?;
    assert!(used()? > baseline);
    // Retain the context so errors from asynchronous field destructors can be
    // observed after the model and its allocator allowance are released.
    std::thread::spawn(move || drop(model)).join().unwrap();
    device.synchronize()?;
    assert_eq!(
        used()?,
        baseline,
        "an idle-worker drop leaked device allocations"
    );
    Ok(())
}
use crate::decode::{commit_block, prefill};

fn identical(a: &Tensor, b: &Tensor, context: &str) -> Result<()> {
    ensure!(a.dims() == b.dims(), "{context}: shape mismatch");
    let error = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    ensure!(error == 0., "{context}: maximum error {error}");
    Ok(())
}

fn identical_prefix(a: &Cache, b: &Cache, len: usize) -> Result<()> {
    ensure!(a.tokens()[..len] == b.tokens()[..len], "token mismatch");
    for (layer, (a, b)) in a.layers.iter().zip(&b.layers).enumerate() {
        let (ak, av) = a.as_ref().context("missing source layer")?;
        let (bk, bv) = b.as_ref().context("missing destination layer")?;
        for (name, a, b) in [("K", ak, bk), ("V", av, bv)] {
            for head in 0..a.dim(0)? {
                identical(
                    &a.get(head)?.narrow(0, 0, len)?,
                    &b.get(head)?.narrow(0, 0, len)?,
                    &format!("layer {layer} head {head} {name}"),
                )?;
            }
        }
    }
    Ok(())
}

// Keep execution shapes fixed to distinguish K/V corruption from differences in
// floating-point arithmetic when a GEMM's batch dimensions change.
fn exercise_storage(model: &Model, prefix: &[u32]) -> Result<()> {
    let b = model.config.block_size;
    let n = prefix.len();
    ensure!(n >= b && n.is_multiple_of(b), "invalid test prefix");
    let mut source = Cache::new(&model.config, n + b)?;
    prefill(model, &prefix[..n - b], &mut source, || false)?;
    commit_block(model, &prefix[n - b..], &mut source)?;
    let query = vec![1; b];
    let expected = model.forward(&query, &mut source, true, None)?.unwrap();

    let mut fork = source.copy_prefix(&model.config, n, n + 3 * b)?;
    identical_prefix(&source, &fork, n)?;
    identical(
        &expected,
        &model.forward(&query, &mut fork, true, None)?.unwrap(),
        "copy with a different head stride",
    )?;
    // Replacing the fork's last committed block must not mutate the source.
    fork.truncate(n - b)?;
    commit_block(model, &vec![2; b], &mut fork)?;
    identical(
        &expected,
        &model.forward(&query, &mut source, true, None)?.unwrap(),
        "source after fork overwrite",
    )?;
    // Restore that block, using the exact same execution shape as the baseline.
    fork.truncate(n - b)?;
    commit_block(model, &prefix[n - b..], &mut fork)?;
    identical_prefix(&source, &fork, n)?;
    identical(
        &expected,
        &model.forward(&query, &mut fork, true, None)?.unwrap(),
        "truncate and recompute",
    )?;
    source.reserve(&model.config, n + 5 * b)?;
    identical_prefix(&source, &fork, n)?;
    identical(
        &expected,
        &model.forward(&query, &mut source, true, None)?.unwrap(),
        "buffer growth",
    )?;
    // Scratch from an interrupted refinement is never a committed cache hit.
    model.forward(&vec![3; b], &mut source, false, None)?;
    source.truncate(n)?;
    ensure!(!source.staged_matches(&vec![3; b]), "stale staged tokens");
    identical(
        &expected,
        &model.forward(&query, &mut source, true, None)?.unwrap(),
        "discarded refinement",
    )?;
    model.device().synchronize()?;
    Ok(())
}

#[test]
fn cache_storage_preserves_values_and_execution() -> Result<()> {
    let model = Model::load(Path::new("tests/fixtures/tiny"), DType::F32, &Device::Cpu)?;
    exercise_storage(&model, &vec![1; 64])
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires CUDA; set MINNOW_CACHE_CHECKPOINT and MINNOW_CACHE_IDS for a guarded full-model check"]
fn cuda_cache_storage_preserves_values_and_execution() -> Result<()> {
    let checkpoint =
        std::env::var("MINNOW_CACHE_CHECKPOINT").unwrap_or_else(|_| "tests/fixtures/tiny".into());
    let prefix = match std::env::var("MINNOW_CACHE_IDS") {
        Ok(path) => {
            let mut ids: Vec<u32> = serde_json::from_slice(&std::fs::read(path)?)?;
            ids.truncate(ids.len() / 32 * 32);
            ids
        }
        Err(_) => vec![1; 64],
    };
    let model = Model::load(Path::new(&checkpoint), DType::BF16, &Device::new_cuda(0)?)?;
    exercise_storage(&model, &prefix)
}
