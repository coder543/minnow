# Checkpoint loading

Measurements on September 8, 2026, NVIDIA GB10, CUDA 13.0, Linux aarch64,
local NVMe storage, release build. Results are in [loading.json](loading.json).
All checkpoints use BF16 activations; INT8 and NVFP4 quantize routed experts.

## Changes

Normal loading checks structure, bounds, encodings, and shapes without hashing
weights. `minnow --model CHECKPOINT.mnw validate` verifies the manifest and all
weight/scale checksums separately, using bounded host memory and no CUDA context.
`validate --reference DIRECTORY` retains the numerical-comparison workflow.

The reader issues up to sixteen concurrent expert reads, combines adjacent code
and scale regions, and reuses its buffers across layers. Staging is bounded at
about 136 MiB, including the separate streaming buffer for large tensors.
Matching-dtype uploads fill final CUDA allocations directly, removing temporary
device tensors, copy kernels, and destination zeroing. Small dtype conversions
retain Candle's conversion semantics. One weight copy remains resident per
instance. Checkpoint files are never mapped.

## Loading time

Each row reports three separate process loads, with one model resident at a time.
Times include model construction, file reads, allocation, and upload; they exclude
CUDA initialization, tokenizer setup, inference, and unloading. No run is discarded.
GB/s uses decimal bytes and the complete container file size.

| Checkpoint | Median | Range | GB/s |
| --- | ---: | ---: | ---: |
| Mini BF16 | 6.86 s | 6.79–7.01 s | 4.74 |
| Mini INT8 | 4.03 s | 4.00–4.03 s | 4.34 |
| Mini NVFP4 | 3.30 s | 2.74–3.30 s | 3.19 |
| Flash NVFP4 | 13.85 s | 13.50–13.88 s | 4.49 |
| Flash INT8 | 19.73 s | 19.69–26.28 s | 5.45 |

## Why the old loader was slow

Identical Nsight Systems settings measured Mini INT8 at 21.64 seconds before
and 4.18 seconds after, a **5.18×** improvement. The old load spent 10.09 seconds
on serialized reads, 9.15 seconds hashing payloads, and 1.65 seconds consuming
chunks through temporary CUDA tensors. The final load spends 2.68 seconds in
read batches and 0.68 seconds consuming chunks. CUDA allocation calls fall from
29,918 to 373, and kernel launches from 29,583 to 19; the remaining kernels perform
dtype conversions.

A read-only, direct-I/O storage test reached **13.13 GB/s** with 1 MiB requests,
queue depth 16, and an 8 GiB extent. Complete model loading remains below raw
storage throughput: reads occur in tensor batches, followed by uploads, and
model allocation/construction also takes time. The loader's `read_seconds`
metric sums latency across concurrent workers; `read_wall_seconds` measures
elapsed read-batch time. Do not add worker latency to wall-clock stages.

The Flash INT8 profile takes 20.04 seconds: read batches account for 11.51
seconds (9.34 GB/s of weight payload), chunk consumption for 2.63 seconds,
and `cuMemAllocAsync` for 5.75 seconds. Allocation is the largest remaining
non-I/O cost. The same minimal-request llama-swap measurement used previously
falls from 98.77 to 21.94 seconds, including startup and the first response.

## Reproduce

```sh
cargo build --release --features cuda --bin minnow --example bench_load
target/release/examples/bench_load models/llada2.2-mini-int8-packed.mnw
minnow --model models/llada2.2-mini-int8-packed.mnw validate
```

The benchmark uses 4 GiB of host load headroom, matching the measured deployment.
Run measurements separately from serving and compilation. `RAYON_NUM_THREADS`
can limit the existing Rayon pool; the reader uses at most sixteen workers.
The eight-worker comparison loaded Mini INT8 in 4.83 seconds versus 4.07 with
sixteen, both with persistent staging and direct uploads.

Validation covers byte ordering across direct-read chunks, CPU and CUDA dtype
conversion, unaligned host bytes, incomplete uploads, unchanged fixture forwards,
manifest/weight/scale corruption, and the checksum command with CUDA hidden.
Real API tests pass streaming, prefill progress, tools and tool-result follow-up,
UI assets, and cancellation after loading the new buffers.
