# Quantization performance

Measured September 8, 2026 on NVIDIA GB10 with CUDA 13, a release build, and
one resident model at a time. Only routed experts are quantized. INT8 uses
BF16 activations; NVFP4 uses native E2M1 matrix operands with E4M3 block scales
and FP32 accumulation. Other layers retain source precision.

Each case discards one warmup and measures three sequential requests, with
prefix reuse disabled. Prefill batches contain up to 4,096 tokens, attention
tiles up to 1,024 queries, and allocator scratch retention is 2 GiB. Generation
is greedy: threshold 0.5, editing threshold 0, 16 maximum post-steps, normal EOS,
and a 2,048-token cap. Loading, tokenization, and admission are excluded.
BF16/INT8 baselines are retained from the preceding build; NVFP4 uses the new
kernels. [Measurement data](quantization-performance.json) records checkpoint
identities, settings, repetitions, work counts, and memory observations.
The [Nsight profile](nvfp4-profile.md) separates attention, native expert GEMMs,
weight traffic, and dispatch costs, with priorities for further optimization.

## Cold prefill

| Prompt tokens | Mini BF16 | Mini INT8 | Mini NVFP4 | Flash NVFP4 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 4,345 | 4,991 | 9,108 | 1,840 |
| 2048 | 7,425 | 6,837 | 11,420 | 2,795 |
| 4096 | 7,762 | 6,539 | 10,291 | 2,765 |
| 8192 | 6,413 | 5,569 | 7,972 | 2,237 |

Rates are total prompt tokens divided by summed prefill seconds. All requests
recompute the complete prompt using the same benchmark token sequence.

## Useful decode

| Prompt | Mini BF16 | Mini INT8 | Mini NVFP4 | Flash NVFP4 |
| --- | ---: | ---: | ---: | ---: |
| What is the LHC? | 41.9 | 60.3 | 92.2 | 22.3 |
| React TypeScript example | 113.7 | 266.4 | 192.9 | 47.4 |
| Fellowship characters/backstories | 39.2 | 56.8 | 83.0 | 22.0 |

These are non-special generated tokens/s, including refinement and commit costs.
They are not repeated predicted positions. Quantization changes the response
and its refinement workload, so these are not identical-work speedups. The INT8
React response reached the output cap; all NVFP4 responses stopped at EOS.

| NVFP4 response | Mini tokens / steps per block | Flash tokens / steps per block |
| --- | ---: | ---: |
| LHC | 352 / 18.50 | 316 / 16.27 |
| React | 672 / 9.27 | 425 / 8.36 |
| Fellowship | 750 / 21.08 | 842 / 17.39 |

These prompts measure throughput, not answer quality. Both BF16 and quantized
outputs contain factual errors and repetition; the mini NVFP4 React example
also contains an undefined handler. Kernel/reference agreement does not establish
negligible quantization loss. A representative quality evaluation remains needed.

## Native kernel optimizations

Both prefill and decode execute native block-scaled FP4 MMA. Disassembly confirms
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`; there is no BF16 expansion of weights.
Weights are stored directly in register-fragment order. Production uses 16- or
32-row tiles selected from a 16/32/64/128 sweep. Specialized activation quantizers
retain input pairs in registers across the scale reduction. Gate/up quantize each
original token once and index it across expert assignments, eliminating repeated
quantization and the duplicated BF16 input gather.

That last optimization, together with quantizer/tile tuning, raised mini 4K
prefill from 8,909 to 10,291 tok/s while preserving all three NVFP4 response texts
and refinement counts. Decode gains from this tuning were small. Flash decode
still requires many refinements; native FP4 acceleration alone does not remove
that cost.

The synthetic benchmark uses 48 experts and includes quantization, allocation,
and synchronization. At six rows per expert, mini gate/up takes 0.118 ms with
NVFP4 versus 0.462 ms with BF16; at 128 rows, 0.374 versus 0.630 ms. Flash gate/up
takes 0.574 versus 1.842 ms at six rows, and 1.198 versus 2.152 ms at 128 rows.
These isolate projection work, not whole-response throughput.

## Memory and validation

| NVFP4 model | Weight payload | Peak system-memory growth |
| --- | ---: | ---: |
| mini-nvfp4 | 9.79 GiB | 12.35 GiB |
| flash-nvfp4 | 57.96 GiB | 63.26 GiB |

Peak growth includes loading, K/V, activations, and workspaces relative to the
unloaded system baseline. These are unified-memory observations, not discrete
GPU VRAM measurements or full-context capacity tests.

CUDA tests compare activation codes/scales exactly against a scalar encoder at
all mini/flash projection widths. Native GEMM is checked against independently
dequantized operands across uneven/repeated expert assignments and all row tiles.
Quantizing before routing matches quantizing gathered rows exactly. Container
checks cover scale metadata, checksums, lossless copying, and rejected
requantization. See [reproduction commands](validation.md).
