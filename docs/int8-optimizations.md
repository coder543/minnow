# INT8 execution and CPU fallback

Measurements on NVIDIA GB10 (SM121), CUDA 13, September 2026. INT8 is W8A16:
packed signed 8-bit expert weights with FP16 scales per 128 weights, BF16
activations and tensor-core operands, and FP32 accumulation. This change does
not requantize checkpoints or quantize activations.

## Execution changes

- GPU routing for individual 32-token decode blocks, reusing the compact routing
  and expert-mixing machinery used by NVFP4. No per-layer router download or
  host-created GEMM descriptors on this path.
- Indexed gate/up inputs during prefill and decode, avoiding the expanded input
  gather. Prefill and multi-block batches still compute routing on the host.
- Fused gate/up plus SiLU, with intermediate projection results in shared memory.
  Gate/up BF16 rounding, SiLU BF16 rounding, and the final BF16 multiplication
  retain the previous operation order.
- Four independent K16 weight-fragment loads issued ahead of their matrix
  operations. Dequantized weights stay in registers; there is no expanded GPU
  weight allocation. Source-row addresses are computed outside the K loop.
- A 16-row default for packed INT8. Sweeps of 16/32/64/128 rows and deeper
  prefetching favored the smaller configuration on this GPU. Larger tiles
  increased latency; fewer repeated weight reads alone did not make them faster.

`--host-routing` and `--unfused-activation` retain comparison paths.
`MINNOW_QUANT_TILE_ROWS=16|32|64|128` selects host-routed tiles for experiments;
compact device routing uses 16-row tiles. Row-major INT8 retains its 32-row
baseline and diagnostic WMMA path.

Nsight Compute on the previous packed kernel reported 87–95% long-scoreboard
stalls and only 4–11% tensor-pipe activity on the measured projection fixtures.
These are memory-dependency stalls, not evidence of compute-bound prefill or
proof that decode saturates memory bandwidth. INT8 still has room for improved
load scheduling and matrix tiling.

In five-refinement Nsight Systems captures, device-to-host copies fell from 95
to zero for mini and from 155 to zero for flash. Host-to-device copies fell from
675/1,095 to ten in each capture (the remaining token uploads). This verifies
that the single-block path actually removes the router round trips.

The updated flash-width projection fixture at 128 rows/expert reduced kernel
time from 3.63 to 3.27 ms and long-scoreboard stalls from 88.8% to 80.2%.
At six rows/expert, cold-cache kernel time was essentially unchanged
(1.13 versus 1.15 ms). The end-to-end decode gains include routing, indexing,
and fusion; they should not be attributed to faster standalone GEMMs alone.

## Controlled whole-model measurements

Baseline: commit `0a2366f`. Both versions use BF16 activations, default block
FlashAttention, a 4,096-token prefill chunk, and a 512 MiB workspace cache. Models
run sequentially. Prefill excludes loading and the vocabulary projection, with
two warmups followed by three measured repetitions. Refinement includes the
vocabulary projection and measures ten identical 32-token forwards after a
warmup. Refinement milliseconds are **not useful generated tokens/second**.

| Model / weights | Prefill 512 tok/s | Prefill 2,048 tok/s | Prefill 4,096 tok/s | Prefill 8,192 tok/s | Refinement at 1,024-token prefix, ms |
|---|---:|---:|---:|---:|---:|
| Mini BF16 | 4,460 | 8,646 | 10,133 | 9,865 | 35.40 |
| Mini INT8 before | 5,692 | 8,071 | 8,484 | 8,178 | 25.95 |
| Mini INT8 after | 6,457 | 9,305 | 9,759 | 9,348 | 22.19 |
| Mini NVFP4 | 10,370 | 15,520 | 15,810 | 14,802 | 16.52 |
| Flash INT8 before | 1,019 | 1,590 | 1,751 | 1,714 | 131.69 |
| Flash INT8 after | 1,042 | 1,653 | 1,853 | 1,799 | 121.45 |
| Flash NVFP4 | 2,010 | 3,513 | 3,870 | 3,677 | 78.99 |

Mini INT8 gains about 15% prefill throughput and 17% fixed-work refinement
throughput at these 4,096/1,024-token settings. Flash gains about 6% and 8%.
NVFP4 remains faster. Mini BF16 still slightly outperforms INT8 for large
prefills. Flash BF16 weights exceed this machine's memory; no full-model Flash
BF16 result is claimed.

A whole-model 4,096-token prefill sweep also confirmed the 16-row choice:
mini reached 9,759/9,238/8,855 tok/s with 16/32/64 rows; flash reached
1,853/1,738/1,633 tok/s. These tests include the fused indexed gate/up kernel,
not just standalone projection fixtures.

Reproduce with the same checkpoint and explicit workspace size:

```sh
minnow --model models/mini-int8.mnw --workspace-cache-mib 512 \
  prefill-bench --input tests/prefill_benchmark_ids.json \
  --tokens 512,2048,4096,8192 --iterations 3

minnow --model models/mini-int8.mnw --workspace-cache-mib 512 \
  bench --input tests/prefill_benchmark_ids.json --cached-only \
  --prefix-blocks 0,32,128 --iterations 10
```

`examples/bench_expert_formats.rs` compares BF16, INT8 and NVFP4 projections at
both mini and flash widths, with 48 distinct expert allocations and 6/32/128 rows
per expert. It holds only one format at a time. This isolates projection cost;
it does not measure routing, denoising, or model quality.

## Useful response throughput

Three prompts from `tests/decode_natural_cases.json`, with the output cap set to
512 tokens: one discarded warmup and one measured response per prompt. Useful
text-token rates exclude prefill. Every before/after pair has identical output
token IDs and per-block refinement counts; the optimization changes execution,
not this measured output. React and Fellowship hit the cap, so those rows measure
512-token continuations, not completed answers.

| Model | Prompt | Useful tokens | Useful tok/s before | Useful tok/s after | Refinements/block |
|---|---|---:|---:|---:|---:|
| Mini INT8 | LHC | 190 | 64.40 | 75.59 | 17.00 |
| Mini INT8 | React TypeScript | 512 (capped) | 202.60 | 239.20 | 6.06 |
| Mini INT8 | Fellowship | 512 (capped) | 50.85 | 60.10 | 23.71 |
| Flash INT8 | LHC | 358 | 18.78 | 20.36 | 12.33 |
| Flash INT8 | React TypeScript | 512 (capped) | 43.66 | 48.09 | 5.47 |
| Flash INT8 | Fellowship | 512 (capped) | 14.08 | 15.46 | 16.59 |

These few deterministic examples establish execution equivalence and workload
sensitivity, not a broad model-quality evaluation. Repeated timing samples and
per-prompt results are in [the measurement data](int8-measurements.json).

## Attention backend

Nsight Systems captures confirm `minnow_flash_block32` launches for mini BF16,
mini INT8, mini NVFP4, flash INT8, and flash NVFP4. There are 20 launches per mini
forward and 32 per flash forward. Weight quantization does not disable flash
attention. CPU, FP32, unsupported shapes, and explicit materialized-attention
options use the materialized implementation.

Startup logs and `/health.attention_backend` expose the selected attention
backend, as does `props["minnow"]["attention_backend"]` from `/props`.
`prefill-bench.flash_attention` now reports the actual selection, including CPU
and FP32 fallback, rather than only whether the option was requested.

## CPU execution

A CPU-only build requires no NVIDIA libraries. Device and dtype defaults select
CPU/FP32 when CUDA is unavailable. Explicit `--device cuda` still reports an
initialization error instead of changing backends. CUDA builds link NVIDIA
libraries and require them even if no GPU is visible; deploy the CPU build on
systems without those libraries.

Candle 0.11's CPU GEMM does not support BF16. Minnow rejects explicit CPU/BF16
before loading weights and defaults to FP32. Quantized expert codes and scales
remain packed. The CPU borrows that storage, decodes directly into one selected
expert's FP32 GEMM buffer in parallel, and drops it before evaluating the next
expert. It does not copy packed experts or unpack their layouts into temporary
code/scale arrays. INT4/INT8 retain BF16 operand rounding; NVFP4 retains activation
quantization in its scalar reference implementation. A persistent Candle worker
pool spans each model forward.

Unquantized weights, activations, and K/V use FP32 on CPU. Budget twice the BF16
size of the unquantized weights, plus expert scratch and runtime memory. CPU
execution is a functional fallback, not a substitute for GPU throughput.
Independent row-major oracle tests cover both INT4/INT8 layouts and NVFP4 at every
mini/flash projection width, repeated experts, and nonzero storage offsets.

Candle has an Apple Metal backend, but minnow does not enable or integrate it.
The pinned Candle backend set is CPU/CUDA/Metal, with no generic AMD/Intel GPU
backend. Porting minnow requires explicit backend work; Candle does not
transparently execute unsupported custom CUDA kernels on another GPU.

On the GB10 ARM CPU, mini INT8 prefill improved from **2.27 to 13.69 tok/s**
for 32 tokens and from **5.86 to 24.32 tok/s** for 256 tokens (two warmups, one
measured repetition, CPU/FP32 for both versions). These are CPU-only prefill
measurements, excluding loading and the output head, not useful decode rates.
