//! Two batches circulate between a parallel direct reader and the upload thread.
//! Ownership of a batch prevents reuse until every consumer has completed.
use super::*;
use rayon::prelude::*;
use std::sync::mpsc;

struct Segment {
    expert: usize,
    scales: bool,
    start: usize,
    bytes: usize,
}
struct Chunk {
    file: usize,
    offset: u64,
    length: usize,
    needed: usize,
    segments: Vec<Segment>,
}
impl Chunk {
    fn new(file: usize, offset: u64, bytes: usize, expert: usize, scales: bool) -> Self {
        let aligned = offset / ALIGN as u64 * ALIGN as u64;
        let start = (offset - aligned) as usize;
        Self {
            file,
            offset: aligned,
            length: (start + bytes).div_ceil(ALIGN) * ALIGN,
            needed: start + bytes,
            segments: vec![Segment {
                expert,
                scales,
                start,
                bytes,
            }],
        }
    }
}

impl WeightLoader {
    pub(super) fn read_parts(
        &self,
        name: &str,
        parts: &[Location],
        mut consume: impl FnMut(usize, bool, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let mut chunks = Vec::new();
        for (expert, part) in parts.iter().enumerate() {
            if let Some(s) = &part.scales {
                let start = part.offset.min(s.offset);
                let end = (part.offset + part.bytes as u64).max(s.offset + s.bytes);
                if end - start <= STAGING_BYTES as u64 {
                    let mut chunk =
                        Chunk::new(part.file, start, (end - start) as usize, expert, false);
                    chunk.segments = vec![
                        Segment {
                            expert,
                            scales: false,
                            start: (part.offset - chunk.offset) as usize,
                            bytes: part.bytes,
                        },
                        Segment {
                            expert,
                            scales: true,
                            start: (s.offset - chunk.offset) as usize,
                            bytes: s.bytes as usize,
                        },
                    ];
                    chunks.push(chunk);
                    continue;
                }
            }
            for (scales, offset, bytes) in std::iter::once((false, part.offset, part.bytes)).chain(
                part.scales
                    .iter()
                    .map(|s| (true, s.offset, s.bytes as usize)),
            ) {
                for copied in (0..bytes).step_by(STAGING_BYTES) {
                    chunks.push(Chunk::new(
                        part.file,
                        offset + copied as u64,
                        (bytes - copied).min(STAGING_BYTES),
                        expert,
                        scales,
                    ));
                }
            }
        }
        if chunks.is_empty() {
            return Ok(());
        }
        // A caller may itself occupy a Rayon worker. An owned pool avoids
        // waiting on that blocked caller when its pool has only one worker.
        let readers = self
            .read_pool
            .get_or_init(|| {
                rayon::ThreadPoolBuilder::new()
                    .thread_name(|i| format!("weight-reader-{i}"))
                    .build()
                    .map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| anyhow::anyhow!("creating checkpoint readers: {e}"))?;
        let batch_size = chunks.len().min(64);
        let capacity = chunks.iter().map(|c| c.length).max().unwrap();
        let begun = Instant::now();
        let mut pool = self
            .read_buffers
            .lock()
            .map_err(|_| anyhow::anyhow!("read buffer lock poisoned"))?;
        while pool.len() < batch_size * 2 {
            pool.push(Staging::with_capacity(capacity)?);
        }
        for buffer in pool.iter_mut() {
            if buffer.len < capacity {
                *buffer = Staging::with_capacity(capacity)?;
            }
        }
        let second = pool.split_off(batch_size);
        let first = std::mem::take(&mut *pool);
        self.timings.lock().unwrap().staging += begun.elapsed();
        std::thread::scope(|scope| -> Result<()> {
            let (ready_tx, ready_rx) = mpsc::sync_channel(1);
            let (recycle_tx, recycle_rx) = mpsc::channel();
            let chunks = &chunks;
            let reader = scope.spawn(move || -> Result<()> {
                let mut spare = Some(second);
                let mut batch = first;
                for base in (0..chunks.len()).step_by(batch_size) {
                    let count = batch_size.min(chunks.len() - base);
                    let start = Instant::now();
                    let reads: Vec<Result<Duration>> = readers.install(|| {
                        batch[..count]
                            .par_iter_mut()
                            .enumerate()
                            .map(|(i, buffer)| {
                                let chunk = &chunks[base + i];
                                let started = Instant::now();
                                let got = loop {
                                    match self.files[chunk.file]
                                        .read_at(&mut buffer.bytes()[..chunk.length], chunk.offset)
                                    {
                                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                                            continue;
                                        }
                                        result => {
                                            break result.with_context(|| {
                                                format!("direct read of {name}")
                                            })?;
                                        }
                                    }
                                };
                                ensure!(got >= chunk.needed, "short direct read for {name}");
                                Ok(started.elapsed())
                            })
                            .collect()
                    });
                    {
                        let mut t = self.timings.lock().unwrap();
                        t.read_wall += start.elapsed();
                        for (i, read) in reads.into_iter().enumerate() {
                            t.read += read?;
                            t.reads += 1;
                            t.bytes += chunks[base + i]
                                .segments
                                .iter()
                                .map(|s| s.bytes as u64)
                                .sum::<u64>();
                        }
                    }
                    if ready_tx.send((base, batch)).is_err() {
                        return Ok(());
                    }
                    batch = match spare.take().or_else(|| recycle_rx.recv().ok()) {
                        Some(batch) => batch,
                        None => return Ok(()),
                    };
                }
                // Return ownership to the caller's pool after the final read.
                let _ = ready_tx.send((chunks.len(), batch));
                if let Ok(batch) = recycle_rx.recv() {
                    let _ = ready_tx.send((chunks.len(), batch));
                }
                Ok(())
            });
            let result = (|| -> Result<()> {
                loop {
                    let start = Instant::now();
                    let next = ready_rx.recv();
                    self.timings.lock().unwrap().read_wait += start.elapsed();
                    let Ok((base, mut batch)) = next else {
                        break;
                    };
                    if base == chunks.len() {
                        pool.extend(batch);
                        continue;
                    }
                    for (i, buffer) in batch
                        .iter_mut()
                        .take(batch_size.min(chunks.len() - base))
                        .enumerate()
                    {
                        let start = Instant::now();
                        let bytes = buffer.bytes();
                        for s in &chunks[base + i].segments {
                            consume(s.expert, s.scales, &bytes[s.start..s.start + s.bytes])?;
                        }
                        self.timings.lock().unwrap().consume += start.elapsed();
                    }
                    if let Err(error) = recycle_tx.send(batch) {
                        pool.extend(error.0);
                    }
                }
                Ok(())
            })();
            // Disconnect both channels before joining, including consumer failures.
            drop(ready_rx);
            drop(recycle_tx);
            let read_result = reader
                .join()
                .map_err(|_| anyhow::anyhow!("checkpoint reader panicked"))?;
            result.and(read_result)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_disconnects_on_consumer_failure_and_reports_short_reads() -> Result<()> {
        let dir = crate::weights::tests::Directory::new();
        let path = dir.0.join("model.safetensors");
        let values = Tensor::arange(0u32, 1024, &Device::Cpu)?.to_dtype(DType::F32)?;
        candle_core::safetensors::save(&HashMap::from([("data", values)]), &path)?;
        let loader = WeightLoader::open(&dir.0)?;
        let parts = vec![loader.tensors["data"].clone(); 137];
        let error = loader
            .read_parts("data", &parts, |_, _, _| bail!("consumer failed"))
            .unwrap_err();
        assert!(error.to_string().contains("consumer failed"));
        let mut seen = Vec::new();
        let caller = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
        caller.install(|| {
            loader.read_parts("data", &parts, |i, scales, bytes| {
                assert!(!scales);
                assert_eq!(bytes.len(), 4096);
                seen.push(i);
                Ok(())
            })
        })?;
        assert_eq!(seen, (0..137).collect::<Vec<_>>());
        assert_eq!(loader.read_buffers.lock().unwrap().len(), 2 * 64);
        File::options().write(true).open(path)?.set_len(0)?;
        let error = loader
            .read_parts("data", &parts, |_, _, _| Ok(()))
            .unwrap_err();
        assert!(error.to_string().contains("short direct read"));
        Ok(())
    }
}
