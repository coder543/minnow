//! Allocate final CUDA weights from metadata while the caller reads/uploads.
use super::*;
use std::{
    collections::VecDeque,
    sync::{Arc, Condvar},
    thread::JoinHandle,
};

type Key = (String, bool);
fn model_order(name: &str) -> (usize, usize) {
    if name == "model.word_embeddings.weight" {
        return (0, 0);
    }
    if let Some(rest) = name.strip_prefix("model.layers.")
        && let Some((layer, _)) = rest.split_once('.')
        && let Ok(layer) = layer.parse()
    {
        return (1, layer);
    }
    (2, 0)
}
#[derive(Clone)]
struct Spec {
    key: Key,
    count: usize,
    dtype: DType,
    order: (usize, u64),
}
#[derive(Default)]
struct State {
    ready: HashMap<Key, WeightBuffer>,
    ready_bytes: usize,
    taken: std::collections::HashSet<Key>,
    requested: Option<Key>,
    error: Option<String>,
    stopped: bool,
    done: bool,
}
pub(super) struct Allocations {
    specs: HashMap<Key, Spec>,
    device: Device,
    state: Arc<(Mutex<State>, Condvar)>,
    worker: Option<JoinHandle<()>>,
}
impl Allocations {
    pub(super) fn start(
        tensors: &HashMap<String, Location>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut specs: HashMap<Key, Spec> = HashMap::new();
        for (name, part) in tensors {
            // Router biases are read on the CPU and separately uploaded by Moe.
            if name.ends_with(".gate.expert_bias") {
                continue;
            }
            let packed = name
                .rsplit_once(".experts.")
                .and_then(|(prefix, suffix)| {
                    let (expert, projection) = suffix.split_once('.')?;
                    expert.parse::<usize>().ok()?;
                    Some(format!("{prefix}.experts.{projection}"))
                })
                .unwrap_or_else(|| name.clone());
            let mut add = |scales: bool, count: usize, dtype: DType| -> Result<()> {
                let key = (packed.clone(), scales);
                let spec = specs.entry(key.clone()).or_insert(Spec {
                    key,
                    count: 0,
                    dtype,
                    order: (part.file, part.offset),
                });
                ensure!(spec.dtype == dtype, "mixed allocation types for {name}");
                spec.count = spec
                    .count
                    .checked_add(count)
                    .context("weight allocation size overflow")?;
                spec.order = spec.order.min((part.file, part.offset));
                Ok(())
            };
            if part.encoding.quantized() {
                add(false, part.bytes, DType::U8)?;
                let s = part
                    .scales
                    .as_ref()
                    .context("missing quantization scales")?;
                let dtype = if part.encoding == Encoding::Nvfp4 {
                    DType::U8
                } else {
                    DType::F16
                };
                add(true, s.bytes as usize / dtype.size_in_bytes(), dtype)?;
            } else {
                add(
                    false,
                    part.bytes / part.dtype.size_in_bytes(),
                    if name.ends_with(".gate.weight") {
                        DType::F32
                    } else {
                        dtype
                    },
                )?;
            }
        }
        let mut ordered: Vec<_> = specs.values().cloned().collect();
        ordered.sort_by(|a, b| {
            model_order(&a.key.0)
                .cmp(&model_order(&b.key.0))
                .then(a.order.cmp(&b.order))
                .then(a.key.cmp(&b.key))
        });
        let mut pending: VecDeque<_> = ordered.into();
        let state = Arc::new((Mutex::new(State::default()), Condvar::new()));
        let shared = state.clone();
        let allocation_device = device.clone();
        let Device::Cuda(cuda) = device else {
            bail!("CUDA allocation requires CUDA device")
        };
        let stream = cuda.cuda_stream().context().new_stream()?;
        let worker = std::thread::Builder::new()
            .name("weight-allocator".into())
            .spawn(move || {
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                        let started = Instant::now();
                        let mut allocation_time = Duration::ZERO;
                        let mut bytes = 0usize;
                        while !pending.is_empty() {
                            let spec = {
                                let mut state = shared.0.lock().unwrap();
                                // Unbounded allocation can delay uploads on another stream.
                                // Stay ahead without monopolizing the driver for a whole model.
                                while !state.stopped
                                    && state.requested.is_none()
                                    && state.ready_bytes >= 2 * 1024 * 1024 * 1024
                                {
                                    state = shared.1.wait(state).unwrap();
                                }
                                if state.stopped {
                                    return Ok(());
                                }
                                let index = state
                                    .requested
                                    .take()
                                    .and_then(|key| pending.iter().position(|s| s.key == key))
                                    .unwrap_or(0);
                                pending.remove(index).unwrap()
                            };
                            let start = Instant::now();
                            let buffer = WeightBuffer::on_stream(
                                spec.count,
                                spec.dtype,
                                &allocation_device,
                                &stream,
                            )?;
                            // on_stream synchronizes and transfers ownership to the
                            // inference stream before making the storage available.
                            allocation_time += start.elapsed();
                            bytes += spec.count * spec.dtype.size_in_bytes();
                            {
                                let mut state = shared.0.lock().unwrap();
                                state.ready_bytes += spec.count * spec.dtype.size_in_bytes();
                                state.ready.insert(spec.key, buffer);
                            }
                            shared.1.notify_all();
                        }
                        tracing::info!(
                            bytes,
                            allocation_seconds = allocation_time.as_secs_f64(),
                            wall_seconds = started.elapsed().as_secs_f64(),
                            "checkpoint allocation timings"
                        );
                        Ok(())
                    }));
                let mut state = shared.0.lock().unwrap();
                state.error = match result {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => Some(format!("{error:#}")),
                    Err(_) => Some("checkpoint allocator panicked".into()),
                };
                state.done = true;
                shared.1.notify_all();
            })?;
        Ok(Self {
            specs,
            device: device.clone(),
            state,
            worker: Some(worker),
        })
    }
    pub(super) fn take(
        &self,
        name: &str,
        scales: bool,
        count: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<WeightBuffer> {
        let key = (name.to_owned(), scales);
        let spec = self
            .specs
            .get(&key)
            .with_context(|| format!("missing planned allocation for {name}"))?;
        ensure!(
            spec.count == count && spec.dtype == dtype && self.device.same_device(device),
            "allocation mismatch for {name}"
        );
        let start = Instant::now();
        let mut state = self.state.0.lock().unwrap();
        ensure!(
            !state.taken.contains(&key),
            "weight allocation already consumed: {name}"
        );
        loop {
            if let Some(buffer) = state.ready.remove(&key) {
                state.taken.insert(key.clone());
                state.ready_bytes -= count * dtype.size_in_bytes();
                self.state.1.notify_all();
                tracing::debug!(
                    name,
                    wait_seconds = start.elapsed().as_secs_f64(),
                    "weight allocation ready"
                );
                return Ok(buffer);
            }
            if let Some(error) = &state.error {
                bail!("allocating {name}: {error}");
            }
            ensure!(!state.done, "weight allocation already consumed: {name}");
            state.requested = Some(key.clone());
            self.state.1.notify_all();
            state = self.state.1.wait(state).unwrap();
        }
    }
}
impl Drop for Allocations {
    fn drop(&mut self) {
        self.state.0.lock().unwrap().stopped = true;
        self.state.1.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a CUDA device"]
    fn allocation_handoff_preserves_packed_weights_and_rejects_mismatches() -> Result<()> {
        let dir = crate::weights::tests::Directory::new();
        let device = Device::new_cuda(0)?;
        let mut weights = HashMap::new();
        for e in 0..137 {
            weights.insert(
                format!("mlp.experts.{e}.gate_proj.weight"),
                Tensor::from_vec(
                    (0..15)
                        .map(|i| (e * 100 + i) as f32 / 32.0)
                        .collect::<Vec<_>>(),
                    (3, 5),
                    &Device::Cpu,
                )?,
            );
        }
        weights.insert(
            "mlp.gate.weight".into(),
            Tensor::from_slice(&[1.234567f32; 137], (137, 1), &Device::Cpu)?,
        );
        candle_core::safetensors::save(&weights, dir.0.join("model.safetensors"))?;
        let mut loader = WeightLoader::open(&dir.0)?;
        loader.start_allocations(DType::BF16, &device)?;
        let allocations = loader.allocations.as_ref().unwrap();
        assert!(
            allocations
                .take("mlp.gate.weight", false, 137, DType::BF16, &device)
                .is_err()
        );
        let actual = loader.load(
            "mlp.experts.gate_proj.weight",
            (137, 3, 5).into(),
            DType::BF16,
            &device,
        )?;
        let expected = Tensor::stack(
            &(0..137)
                .map(|e| weights[&format!("mlp.experts.{e}.gate_proj.weight")].clone())
                .collect::<Vec<_>>(),
            0,
        )?
        .to_dtype(DType::BF16)?;
        assert_eq!(
            actual.flatten_all()?.to_vec1::<half::bf16>()?,
            expected.flatten_all()?.to_vec1::<half::bf16>()?
        );
        assert_eq!(
            loader
                .load("mlp.gate.weight", (137, 1).into(), DType::F32, &device)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![1.234567f32; 137]
        );
        assert!(
            allocations
                .take("mlp.gate.weight", false, 137, DType::F32, &device)
                .is_err()
        );
        drop(loader);
        let Device::Cuda(cuda) = &device else {
            unreachable!()
        };
        assert!(!cuda.cuda_stream().context().is_in_multi_stream_mode());
        // Dropping an in-flight plan cancels and joins its worker, releasing unclaimed storage.
        let mut loader = WeightLoader::open(&dir.0)?;
        loader.start_allocations(DType::BF16, &device)?;
        drop(loader);
        device.synchronize()?;
        Ok(())
    }
}
