# NVFP4 optimization follow-up

Measured on NVIDIA GB10, CUDA 13, September 8, 2026. This follows the
[Nsight bottleneck analysis](nvfp4-profile.md). Only routed experts use NVFP4;
other weights remain BF16. Checkpoint contents and quantization are unchanged.

## Changes

- Single-block routing and compact GEMM descriptors are generated in one
  GPU kernel. The 256 token/expert assignments stay on device, including their
  FP32 mixture weights. Empty expert tiles exit immediately.
- Native gate/up GEMMs and SiLU/multiply share one launch and one output buffer.
  The intermediate BF16 rounding points are retained. This is the default for
  NVFP4 prefill and decode, including the host-routing fallback.
- Block FlashAttention specializes FlashAttention/CUTLASS for the
  model's block-causal mask. Score and probability tiles stay on chip; the kernel
  reads the existing strided K/V cache and writes token-major output directly.
  Scratch grows linearly with query count rather than query count times context.

All three changes are enabled by default on supported CUDA shapes.
`--materialized-attention` selects the preceding attention path. GPU routing
currently supports the 256-expert/top-8/capacity-48 single-block shape used by
both mini and flash. Larger batches use host routing. None of these changes creates another weight set or changes the packed
checkpoint format.

## Mini throughput

Rates are useful tokens/s for decode and complete prompt tokens/s for prefill.
The [measurement data](nvfp4-optimizations.json) retains repetitions, work counts,
response hashes, checkpoint identity, and numerical comparisons. The baseline
is the preceding NVFP4 build; the materialized column isolates routing/fusion.

| Prefill tokens | Previous NVFP4 | Routing/fusion, materialized attention | New default |
| ---: | ---: | ---: | ---: |
| 512 | 9,108 | 8,775 | 9,572 |
| 2,048 | 11,420 | 11,578 | 15,211 |
| 4,096 | 10,291 | 10,442 | 15,678 |
| 8,192 | 7,972 | 8,068 | 14,704 |

| Decode prompt | Previous NVFP4 | Routing/fusion, materialized attention | New default |
| --- | ---: | ---: | ---: |
| lhc | 92.2 | 96.3 | 95.3 |
| react | 192.9 | 202.3 | 230.5 |
| fellowship | 83.0 | 85.4 | 78.2 |

Routing/fusion preserves all three response texts and step counts, improving
useful decode by 2.9–4.9%. Default attention changes the denoising trajectory:
LHC produces 215 tokens at 17.38 steps/block, React 1,487 at 8.36, and Fellowship
745 at 22.76. In particular, Fellowship requires more refinements and its useful
rate falls. These are observed response rates, not identical-work speedups.

The isolated attention fixture falls from 7.67 to 0.93 ms at 4K queries and from
26.19 to 3.38 ms at 8K. At 32 queries over a 4K prefix, it rises from 0.067 to
0.101 ms; this kernel primarily benefits prefill. All fixtures use mini head
counts, include output allocation/synchronization, and exclude input creation.

## Updated prefill profile

A fresh Nsight Systems capture of 4K mini prefill takes 259.8 ms. Attention is
now 9.6% of GPU kernel time, down from 40.4% in the preceding profile. Native
expert GEMMs account for 42.1%; expert mixture reduction takes another 7.5%.
Kernel/copy/set activity occupies 73.0% of the GPU timeline, leaving 70.2 ms in
gaps. GPU routing currently accelerates single-block decode only, so prefill's
host routing and descriptor preparation remain candidates for reducing gaps.

The CUDA API table contains long copy calls that wait for preceding work; those
API durations are not physical transfer times. Actual GPU copy activity totals
1.74 ms in this capture. These are Systems timings, not new whole-model hardware
counter evidence of a purely compute- or bandwidth-bound phase.

The bounded 4K attention fixture also runs in 0.925 ms under Nsight Compute,
with 64.2% tensor-pipe active cycles and 25.3 MB of system-memory fills
(27.3 GB/s). This supports a computation-side limit for this attention fixture,
not an external-memory bandwidth limit. Tensor-pipe active cycles are not a
percentage of peak FLOPs, and the fixture does not represent the whole model.

## Flash spot check

One discarded warmup and one measured request per case, with the same settings
as mini. This is a cross-model check, not a three-repetition benchmark.

| Prefill tokens | Previous NVFP4 | New default |
| ---: | ---: | ---: |
| 512 | 1,840 | 2,030 |
| 2,048 | 2,795 | 3,539 |
| 4,096 | 2,765 | 3,932 |
| 8,192 | 2,237 | 3,733 |

Useful decode is 28.0 tok/s for LHC (350 tokens, 13.17 steps/block), 94.1 for
React (2,048 tokens, 4.54 steps/block), and 22.6 for Fellowship (808 tokens,
17.30 steps/block). All measured texts match their warmup. React reaches the
output cap, so its rate is not evidence of faster task completion. Flash shares
the routing shape supported by the new GPU route; its wider projection shapes
also use the fused native kernels.

## Numerical behavior

Routing IDs, mixture weights, projections, and final expert mixtures match the
previous paths bit for bit in CUDA tests. Gate/up fusion is checked with distinct
projection weights, uneven assignments, tail columns, every 16/32/64/128 row tile,
and all mini/flash projection widths. Full-response materialized-attention comparisons
also preserve the previous text and refinement counts on all three prompts.

Fused attention retains the BF16 score and scaled-score rounding points but
uses online softmax. Its probability rounding and accumulation order differ
from materialized attention. Unit tests check the block mask, grouped query
heads, cache strides, query tails, full-context offsets, and consistency between
bulk and individual-block evaluation. This establishes kernel correctness within
numerical tolerances, not equal answer quality.

A two-pass prototype that rounded normalized probabilities reduced local RMS
error from approximately 1e-4 to 1e-7. It still did not restore identical model
outputs: in a trained-input trace, layer 0 matched exactly and the first attention
difference appeared in layer 1 at 1.1e-6 RMS, then grew through subsequent layers.
That more expensive variant was not retained. The default uses online softmax.
Changed response lengths and refinement counts must not be interpreted as
identical-work decode speedups; these throughput checks do not measure answer
quality.

## Reproduction

Build before timing. Run large checks sequentially under the memory guard with
budgets appropriate for the model and available host memory. Each benchmark
loads one resident model, discards one warmup, and measures three sequential
requests. Prefix reuse is disabled, generation is greedy, and the output cap is
2,048 tokens. Prefill uses 4,096-token transformer batches and up to 1,024-query
materialized-attention tiles. Allocation retention is 2 GiB.

```sh
cargo build --release --features cuda --bins --examples
python3 scripts/bench_models.py --model models/mini-nvfp4.mnw \
  --report artifacts/default.json
python3 scripts/bench_models.py --model models/mini-nvfp4.mnw \
  --server-arg=--materialized-attention --report artifacts/materialized.json

target/release/examples/bench_attention
target/release/examples/compare_attention models/mini-nvfp4.mnw --trace
```

`compare_attention` alternates backends with one resident checkpoint and fresh
K/V caches, comparing all 32 vocabulary distributions at five fixed prefix
lengths. Its single-forward times include warmup effects and are not throughput
measurements. `--trace` additionally reports the zero-prefix layer errors.

Register prefetching and 32-column native GEMM tiles were also measured. Neither
produced a consistent gain, and several prefill shapes regressed, so neither is
retained. Native expert multiplication still reads packed weights directly into
registers; an effective asynchronous loading pipeline remains future work.

## Validation

CPU/reference and CUDA suites pass, including native FP4 projection/fusion,
GPU routing, and block attention tests. Server checks pass on both attention
backends for Chat Completions, tools, streaming progress/timings, prefix growth,
forking, LRU eviction, and the scripted warm/cold comparisons. Concurrent
requests, late admission, and cancellation pass with the new default. Warm/cold
checks cover their fixed prompts and do not imply universal bitwise invariance
across different GEMM shapes.

Full-model runs were sequential, each using one weight copy and bounded
loading. Flash's peak system-memory growth is 61.5 GiB in this check, including
weights, K/V, and workspace. No additional checkpoint copy is introduced.
