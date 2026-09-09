# Checkpoint loading

Measurements on September 8, 2026, NVIDIA GB10, CUDA 13.0, Linux aarch64,
local NVMe storage, release build. Results are in [loading.json](loading.json).
All checkpoints use BF16 activations; INT8 and NVFP4 quantize routed experts.

## Changes

Normal loading checks structure, bounds, encodings, and shapes without hashing
weights. `minnow --model CHECKPOINT.mnw validate` verifies the manifest and all
weight/scale checksums separately, using bounded host memory and no CUDA context.
`validate --reference DIRECTORY` retains the numerical-comparison workflow.

The loader overlaps three stages: a background allocator creates final weight
storage from checkpoint metadata, a parallel reader fills the next batch, and
the calling thread uploads the current batch. The allocator follows model layer
order, prioritizes requested weights, and pauses after about 2 GiB of unclaimed
final storage (plus one allocation). Allocations are synchronized and their
ownership transferred to the inference stream without copying. The temporary
stream is released after loading, preserving the original inference stream behavior.

Two reusable batches hold up to 64 direct-read chunks each. Chunk size is bounded
at 8 MiB, adjacent expert codes/scales share a read, and buffers grow only as needed.
Maximum host staging is about 1 GiB plus an 8 MiB conversion/validation buffer.
The reader may fill the next batch while allocation or upload is still running;
ownership prevents buffers from being reused before upload completes. Errors
close the pipeline and join its workers, including early consumer failures.

Matching-dtype uploads fill final CUDA allocations directly, removing temporary
device tensors, copy kernels, and destination zeroing. Small dtype conversions
retain Candle's conversion semantics. One weight copy remains resident per
instance. Checkpoint files are never mapped.

## Loading time

Each row reports three separate process loads, with one model resident at a time.
Times include model construction, file reads, allocation, and upload; they exclude
CUDA initialization, tokenizer setup, inference, and unloading. No run is discarded.
GB/s uses decimal bytes and the complete container file size.

| Checkpoint | Previous median | Pipelined median | Range | GB/s |
| --- | ---: | ---: | ---: | ---: |
| Mini BF16 | 6.86 s | 4.82 s | 4.81–4.84 s | 6.75 |
| Mini INT8 | 4.03 s | 2.73 s | 2.70–2.75 s | 6.40 |
| Mini NVFP4 | 3.30 s | 2.14 s | 2.11–2.15 s | 4.93 |
| Flash NVFP4 | 13.85 s | 8.34 s | 8.32–8.36 s | 7.46 |
| Flash INT8 | 19.73 s | 13.36 s | 13.35–13.41 s | 8.04 |

## Profiling

A read-only, direct-I/O storage test reached **13.13 GB/s** with 1 MiB requests,
queue depth 16, and an 8 GiB extent. Complete model loading remains below raw
storage throughput because of tensor boundaries, allocation stalls, and uploads,
even with those stages overlapping. The loader's `read_seconds`
metric sums latency across concurrent workers; `read_wall_seconds` measures
elapsed read-batch time. Both overlap upload/allocation work in the pipeline.
`read_wait_seconds` and `allocation_wait_seconds` measure the calling thread's
waits for ready data and final storage. `consume_seconds` includes uploads,
conversions, scale-value checks, and any CUDA stalls during those operations.
Do not add overlapping stage times or worker latency to obtain total load time.

The pipelined Flash INT8 Nsight capture takes **13.47 seconds**, compared with
20.04 seconds before pipelining. Read batches occupy 10.61 seconds and the
allocation worker spends 5.40 seconds allocating and handing off storage, with
those stages overlapping. The calling thread spends 0.15 seconds waiting for
allocations and 7.50 seconds waiting for reads. Upload/processing calls occupy
5.37 seconds, including remaining CUDA stalls. Allocation latency still varies;
the unprofiled three-run median above is the throughput comparison.

All five checkpoints loaded successfully under the same sequential memory test;
peak whole-system growth was **103.39 GiB**, including Flash INT8 weights and
transient staging. Staging is released before serving requests.

## Reproduce

```sh
cargo build --release --features cuda --bin minnow --example bench_load
target/release/examples/bench_load models/llada2.2-mini-int8.mnw
minnow --model models/llada2.2-mini-int8.mnw validate
```

The benchmark checks target-device capacity for weights and full K/V before loading.
Run measurements separately from serving and compilation.
The reader uses a dedicated Rayon pool for each batch; `RAYON_NUM_THREADS` controls CPU
read concurrency independently of the bounded batch queue. A 16-chunk pipeline
loaded Flash INT8 in 16–17 seconds; deeper read-ahead improved throughput during
CUDA allocation stalls.

Validation covers byte ordering across direct-read chunks, CPU and CUDA dtype
conversion, unaligned host bytes, incomplete uploads, cross-stream ownership,
early pipeline shutdown, short reads, unchanged fixture forwards,
manifest/weight/scale corruption, and the checksum command with CUDA hidden.
Real API tests pass streaming, prefill progress, tools and tool-result follow-up,
UI assets, and cancellation after loading the new buffers.
