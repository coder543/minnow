//! Bounded host decoders yield transformer work to one GPU worker. The worker
//! continuously admits requests and combines ready prefill/refinement segments.
//! Host threads own sampling/RNG and output parsing, never model evaluation.
use super::*;
use crate::model::{Cache, Forward, MAX_FORWARD_TOKENS};
use std::{collections::VecDeque, sync::mpsc as channel, thread::JoinHandle, time::Instant};

#[derive(Default)]
pub(super) struct Metrics {
    forwards: AtomicU64,
    sequences: AtomicU64,
    max_sequences: AtomicU64,
}
impl Metrics {
    pub(super) fn value(&self) -> Value {
        json!({"forward_batches":self.forwards.load(Ordering::Relaxed),
            "sequence_forwards":self.sequences.load(Ordering::Relaxed),
            "max_batch_size":self.max_sequences.load(Ordering::Relaxed)})
    }
}

struct Call {
    tokens: Vec<u32>,
    cache: Cache,
    logits: bool,
    reply: channel::SyncSender<(Cache, Result<Option<candle_core::Tensor>>)>,
}
struct Proxy {
    model: Arc<Model>,
    calls: channel::Sender<Call>,
}
impl Executor for Proxy {
    fn config(&self) -> &crate::config::Config {
        &self.model.config
    }
    fn prefill_chunk_tokens(&self) -> usize {
        self.model.prefill_chunk_tokens()
    }
    fn synchronize(&self) -> Result<()> {
        Ok(self.model.device().synchronize()?)
    }
    fn forward_tokens(
        &self,
        tokens: &[u32],
        cache: &mut Cache,
        logits: bool,
    ) -> Result<Option<candle_core::Tensor>> {
        let replacement = Cache::new(self.config(), cache.capacity())?;
        let cache_value = std::mem::replace(cache, replacement);
        let (reply, rx) = channel::sync_channel(1);
        self.calls
            .send(Call {
                tokens: tokens.to_vec(),
                cache: cache_value,
                logits,
                reply,
            })
            .map_err(|_| anyhow::anyhow!("inference worker stopped"))?;
        let (returned, result) = rx
            .recv()
            .map_err(|_| anyhow::anyhow!("inference worker stopped"))?;
        *cache = returned;
        result
    }
}
struct Done {
    sequence: usize,
    checkout: Checkout,
    result: Reply,
    transport: Transport,
}
fn finish(transport: Transport, result: Reply) {
    match transport {
        Transport::Json(reply) => {
            let _ = reply.send(result);
        }
        Transport::Stream(reply) => {
            if let Err(error) = result {
                let _ = reply.try_send(Ok(Event::default().data(error.value().to_string())));
            }
            let _ = reply.try_send(Ok(Event::default().data("[DONE]")));
        }
    }
}

pub(super) fn run(
    model: Arc<Model>,
    mut pool: CachePool,
    app: App,
    mut jobs: mpsc::Receiver<Option<Job>>,
) {
    let (calls_tx, calls_rx) = channel::channel::<Call>();
    let (done_tx, done_rx) = channel::channel::<Done>();
    let mut handles: Vec<Option<JoinHandle<()>>> = (0..app.info.parallel).map(|_| None).collect();
    let mut pending: Option<Job> = None;
    let mut calls = VecDeque::new();
    let mut stopping = false;
    loop {
        for done in done_rx.try_iter() {
            if done.result.is_ok() {
                pool.checkin(done.checkout);
            } else {
                pool.discard(done.checkout);
            }
            *app.cache_slots.lock().unwrap() = pool.snapshot();
            app.active.lock().unwrap()[done.sequence] = None;
            finish(done.transport, done.result);
            if let Some(handle) = handles[done.sequence].take() {
                let _ = handle.join();
            }
        }
        stopping |= app.stopping.load(Ordering::Acquire);
        let active = handles.iter().filter(|h| h.is_some()).count();
        if stopping {
            if let Some(job) = pending.take() {
                finish(
                    job.transport,
                    Err(ApiError::unavailable("server is stopping")),
                );
            }
            while let Ok(Some(job)) = jobs.try_recv() {
                finish(
                    job.transport,
                    Err(ApiError::unavailable("server is stopping")),
                );
            }
            if active == 0 {
                break;
            }
        } else {
            // One FIFO admission candidate is retained when the K/V budget is
            // occupied. Active slots are never considered for LRU eviction.
            while let Some(sequence) = handles.iter().position(Option::is_none) {
                if pending.is_none() {
                    let next = if handles.iter().all(Option::is_none) && calls.is_empty() {
                        jobs.blocking_recv()
                    } else {
                        jobs.try_recv().ok()
                    };
                    match next {
                        Some(Some(job)) => pending = Some(job),
                        Some(None) => {
                            stopping = true;
                            app.stopping.store(true, Ordering::Release);
                            break;
                        }
                        None => break,
                    }
                }
                let job = pending.as_ref().unwrap();
                if job.transport.closed() {
                    pending = None;
                    continue;
                }
                let b = model.config.block_size;
                let max_tokens = job
                    .request
                    .options
                    .max_tokens
                    .unwrap_or(app.info.max_context - job.request.ids.len());
                let capacity = (job.request.ids.len() + max_tokens).div_ceil(b) * b;
                let empty = max_tokens == 0;
                if !empty && !pool.can_admit(capacity, job.request.cache_prompt) {
                    break;
                }
                let mut job = pending.take().unwrap();
                // A request waiting for K/V remains queued even outside the
                // channel. Release its queue permit only on admission.
                drop(job.queued.take());
                let start = Instant::now();
                let n = job.request.ids.len() / b * b;
                let admission = if empty {
                    Checkout::empty(&model, capacity)
                } else {
                    pool.checkout(
                        &model,
                        &job.request.ids[..n],
                        capacity,
                        job.request.cache_prompt,
                    )
                };
                let mut checkout = match admission {
                    Ok(checkout) => checkout,
                    Err(error) => {
                        finish(
                            job.transport,
                            Err(ApiError::new(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                error.to_string(),
                            )),
                        );
                        continue;
                    }
                };
                *app.cache_slots.lock().unwrap() = pool.snapshot();
                let proxy = Proxy {
                    model: model.clone(),
                    calls: calls_tx.clone(),
                };
                let worker = app.clone();
                let done = done_tx.clone();
                handles[sequence] = Some(
                    std::thread::Builder::new()
                        .name(format!("minnow-decode-{sequence}"))
                        .spawn(move || {
                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    execute(&proxy, &mut checkout, &worker, &job, sequence, start)
                                }))
                                .unwrap_or_else(|_| {
                                    Err(ApiError::new(
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        "decoder panicked",
                                    ))
                                });
                            let _ = done.send(Done {
                                sequence,
                                checkout,
                                result,
                                transport: job.transport,
                            });
                        })
                        .expect("starting bounded decoder thread"),
                );
            }
        }
        calls.extend(calls_rx.try_iter());
        if calls.is_empty() {
            // Short bounded wait lets completions, admissions and shutdown run.
            if let Ok(call) = calls_rx.recv_timeout(Duration::from_millis(1)) {
                calls.push_back(call);
            }
            continue;
        }
        let deadline = Instant::now() + Duration::from_micros(app.info.batch_wait_us);
        while calls.len() < handles.iter().filter(|h| h.is_some()).count() {
            let Some(wait) = deadline.checked_duration_since(Instant::now()) else {
                break;
            };
            match calls_rx.recv_timeout(wait) {
                Ok(call) => calls.push_back(call),
                Err(_) => break,
            }
        }
        let mut batch = Vec::new();
        let mut tokens = 0;
        while calls
            .front()
            .is_some_and(|c| tokens + c.tokens.len() <= MAX_FORWARD_TOKENS)
        {
            let call = calls.pop_front().unwrap();
            tokens += call.tokens.len();
            batch.push(call);
        }
        if batch.is_empty() {
            let call = calls.pop_front().unwrap();
            let _ = call.reply.send((
                call.cache,
                Err(anyhow::anyhow!("request exceeds batch workspace")),
            ));
            continue;
        }
        let result = if stopping {
            Err(anyhow::anyhow!("server is stopping"))
        } else {
            app.batch_metrics.forwards.fetch_add(1, Ordering::Relaxed);
            app.batch_metrics
                .sequences
                .fetch_add(batch.len() as u64, Ordering::Relaxed);
            app.batch_metrics
                .max_sequences
                .fetch_max(batch.len() as u64, Ordering::Relaxed);
            let mut work: Vec<_> = batch
                .iter_mut()
                .map(|call| Forward {
                    tokens: &call.tokens,
                    cache: &mut call.cache,
                    logits: call.logits,
                })
                .collect();
            model.forward_batch(&mut work).and_then(|out| {
                model.device().synchronize()?;
                Ok(out)
            })
        };
        match result {
            Ok(output) => {
                for (call, logits) in batch.into_iter().zip(output) {
                    let _ = call.reply.send((call.cache, Ok(logits)));
                }
            }
            Err(error) => {
                for call in batch {
                    let _ = call
                        .reply
                        .send((call.cache, Err(anyhow::anyhow!(error.to_string()))));
                }
            }
        }
    }
}
