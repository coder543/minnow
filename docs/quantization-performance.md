# Quantized mini/flash measurements on GB10

Measured September 8, 2026 using local-SSD `.mnw` checkpoints, BF16 activations,
FP32 accumulation, one resident model, and the default 4,096-token transformer
prefill batch. INT8 uses 128-weight groups; FP4 uses 32-weight groups. Both use
FP16 scales and the lossless tensor-core fragment layout. Only routed experts
are quantized. See [format details](model-format.md) and [measurement data](quantization-performance.json).

Each case has a discarded warmup and three measured requests. Requests are
sequential (`--parallel 1`) and disable prefix reuse. Generation is greedy with
threshold 0.5, editing threshold 0, 16 maximum post-steps, normal EOS stopping,
and a 2,048-token cap. Loading, tokenization, and admission are excluded from the
reported inference rates. Decode includes all refinements and commit refreshes.
The allocator may retain 2 GiB of unused scratch relative to live allocations.

## Cold prefill

| Complete prompt tokens | Mini BF16 | Mini INT8 | Mini FP4 | Flash FP4 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 4,345 | 4,991 | 5,650 | 1,120 |
| 2,048 | 7,425 | 6,837 | 7,078 | 1,485 |
| 4,096 | 7,762 | 6,539 | 6,646 | 1,495 |
| 8,192 | 6,413 | 5,569 | 5,642 | 1,324 |

Values are tokens/s, computed as total prompt tokens divided by summed prefill
time. All requests recompute the entire prompt. The input is the same committed
benchmark token sequence for each model/precision. The quantized kernels improve
512-token prefill, but at larger batches their dequantization/instruction cost
outweighs the bandwidth reduction. Large-row quantized GEMM remains an optimization
opportunity; weight-only quantization does not make every workload faster.

## Useful decode

| Prompt | Mini BF16 | Mini INT8 | Mini FP4 | Flash FP4 |
| --- | ---: | ---: | ---: | ---: |
| What is the LHC? | 41.9 | 60.3 | 71.2 | 19.4 |
| Write a React TypeScript example | 113.7 | 266.4* | 184.1 | 59.1 |
| Fellowship characters and backstories | 39.2 | 56.8 | 61.3 | 16.8 |

These are generated non-special tokens/s, not repeated predicted positions.
The INT8 React response reached the 2,048-token cap (*); all other responses
ended at EOS. Quantization changes the output and sometimes its difficulty, so
these are workload measurements rather than identical-work kernel speedups.

| Prompt | Mini BF16 tokens / steps per block | Mini INT8 | Mini FP4 | Flash FP4 |
| --- | ---: | ---: | ---: | ---: |
| LHC | 293 / 20.30 | 192 / 18.00 | 193 / 18.43 | 368 / 14.33 |
| React | 1,058 / 8.03 | 2,048 / 4.68 | 884 / 8.00 | 1,785 / 4.93 |
| Fellowship | 750 / 22.52 | 813 / 20.74 | 959 / 24.06 | 820 / 16.26 |

All measured repeats matched their same-configuration warmup text. That does
not imply warm/cold-prefix or cross-batch bitwise invariance; see the separate
[cache arithmetic investigation](prefix-cache.md).

The user reports about **140 useful tokens/s for DiffusionGemma 26B A4B NVFP4**
on this machine. Flash is substantially below that reference across these
prompts. Mini exceeds it on this coding example but falls below it on both prose
prompts. The DiffusionGemma run was not repeated here, and matching its prompts,
sampling, precision, and stopping would be needed for a controlled comparison.
Our custom FP4 storage and BF16 activations are not native NVFP4 execution.

These prompts are throughput probes, not an accuracy benchmark. The outputs
include factual errors and occasional repetition in the BF16 baseline as well
as quantized runs. Tests establish implementation correctness against the
stored/dequantized weights; they do not establish negligible quantization loss
or equal answer quality. Full model quality evaluation remains necessary before
choosing a deployment precision on accuracy grounds.

## Kernel and memory checks

For 48 experts with six assigned rows each, fragment packing reduced a mini
INT8 gate/up GEMM from 0.428 to 0.275 ms and FP4 from 0.202 to 0.175 ms. Flash
INT8 changed from 1.931 to 1.102 ms and FP4 from 0.921 to 0.780 ms. These calls
include allocation and synchronization. They isolate a lossless layout change,
without changing codes, scales, or arithmetic. The 32-row tile outperformed the
64/128-row alternatives on the measured shapes and is the default.

CUDA checks cover both layouts/bit widths, every supported scale-group size,
uneven/repeated expert assignments, output-column and K-tile tails, and numerical
agreement with independently dequantized BF16 operands. Container tests check
mixed-precision execution and exact round-trip repacking. Runtime dequantization
stays in registers; there is no expanded full-weight copy.

| Model | Weight payload GiB | Peak system growth GiB in full benchmark |
| --- | ---: | ---: |
| Mini BF16 | 30.28 | 34.21 |
| Mini INT8 | 16.25 | 19.05 |
| Mini FP4 | 9.79 | 12.40 |
| Flash FP4 | 57.96 | 65.60 |

System growth includes loading, runtime, K/V, and temporary workspaces; it is
measured relative to each unloaded baseline on GB10's shared RAM. Allocator
history and other system allocations also affect it. All four runs completed
within their explicit memory budgets with **zero new swap-out**. These are not
full-context or RTX 3090 measurements. Mini INT8 is the sensible starting point
for a 24 GiB card; its 16.25 GiB weights leave a limited budget for K/V and scratch.

## Allocator allowance

With `--workspace-cache-mib 0`, mini BF16 achieved 7,474 tok/s at 4K prefill and
6,072 at 8K, versus 7,762 and 6,413 with the 2 GiB allowance. Decode changed by
less than 2% on each prompt, and all three outputs matched exactly. This ablation
used one warmup and two measured runs. Its peak system growth was 33.42 GiB
versus 34.21 GiB for the baseline. The benefit is modest prefill improvement;
the allowance is not responsible for the quantized decode gains.

## Reproduce

```sh
cargo build --release --features cuda
python3 scripts/memory_guard.py --report artifacts/bench-mini-int8-memory.json \
  --max-growth-gib 30 --reserve-gib 40 -- \
  python3 scripts/bench_models.py \
  --model ~/models/minnow/llada2.2-mini-int8-packed.mnw \
  --report artifacts/bench-mini-int8.json
```

Run models sequentially. For flash, use the FP4 file and an appropriate explicit
budget (the measured run allowed 78 GiB growth and retained 32 GiB system reserve).
`--prefill-only`/`--decode-only` select parts of the suite. Full text and per-block
records remain in local artifacts; committed JSON contains timings, work counts,
container manifest identities, and the synthetic matrix measurements.
