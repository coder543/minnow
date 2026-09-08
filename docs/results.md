# Validation and performance

Measurements on September 7, 2026 (local time), NVIDIA GB10, CUDA 13.0,
Candle 0.11.0, release build. The oracle used PyTorch 2.10.0+cu130 and
Transformers 5.2.0. PyTorch warns that its packaged architecture range ends at
SM 12.0 while this device is SM 12.1; the reference runs completed successfully.
These are development measurements, not a task-quality benchmark.
Machine-readable timings are in [results.json](results.json).
See the subsequent [prefill and useful generation comparison](throughput-comparison.md)
for sustained prefill measurements and the default Transformers SDPA baseline.
The Transformers generation timings below used eager attention.

## Storage and memory

All 20 checkpoint files were copied to
`~/models/hf/inclusionAI/LLaDA2.2-mini` and verified against source SHA-256 hashes.
Copy plus verification took 142.38 seconds. The original network checkout was
read-only throughout. The runtime uses 8 MiB bounded direct reads with no mmap.

| Measurement | Result |
| --- | ---: |
| Load BF16 from original network share | 279.25 s |
| Load BF16 from local NVMe | 8.36 s |
| Checkpoint BF16 tensor bytes | 30.28 GiB |
| FP32 tensor bytes | 60.56 GiB |
| HTTP generation: peak system-memory growth | 31.88 GiB |
| Native FP32 diagnostic: peak system-memory growth | 62.63 GiB |
| Layer-at-a-time FP32 oracle, 96 tokens: peak growth | 4.80 GiB |
| New swap-out pages during these validations | 0 |

Memory figures are the guard's net whole-system growth from its baseline, using
`MemAvailable`; unrelated allocations and cache reclamation can affect them.
Weight ownership is enforced separately by streaming directly into final buffers,
sharing packed expert views, and holding an exclusive cross-process model lock.
"Mini" has about 16.26 billion total MoE parameters. Sparse activation reduces
computation, but the resident unquantized weight set still occupies 30.28 GiB.

The prior memory exhaustion came from retaining complete CPU and CUDA FP32
models on shared system RAM. That loading path has been removed. A 128K BF16
KV cache itself is 5 GiB; full 128K serving performance has not been measured.

## Numerical checks

The checked-in small FP32 fixture exercises three layers, dense/shared/routed
MLPs, block routing, GQA, partial RoPE, Q/K normalization, and three attention
blocks. Every recorded tensor agrees with the checkpoint classes within `1e-4`,
with exact routing. Cache refinement, final-token commit, edit handling, and
non-convergence tests pass.

For the actual trained weights:

| FP32 comparison | 32 tokens | 96 tokens |
| --- | ---: | ---: |
| Maximum logit error against Transformers | 2.86e-5 | 3.27e-4 |
| Logit RMS error | 3.33e-6 | 6.95e-6 |
| Top-token agreement | 100% | 100% |
| Expert selections, all 19 MoE layers | Exact | Exact |
| Cached versus full: maximum logit error | 0 | 2.74e-4 |

The 96-token case includes 64 committed prefix tokens and a partially masked
third block. Every exported tensor passes `atol=1e-3, rtol=2e-5`; expert IDs are
required to match exactly. Raw hidden features reach magnitude 29,258, with
maximum absolute difference 0.02734, so a purely absolute tolerance is unsuitable.
The final hidden-state relative RMS error is approximately `1e-6`. At the stricter
`atol=1e-3, rtol=1e-5`, one hidden entry fails: -259.648865 versus -259.653564.
These differences remain visible in the saved reports rather than being rounded
away. FP32 oracle and runtime never resided together.

The fused SiLU/multiply kernel is bitwise equal to the explicit FP32 activation
path for every finite BF16 input in the CUDA test. All intermediate tensors and
logits of the actual 32-token BF16 checkpoint forward also remain bitwise equal
when enabling fusion. The kernel rounds the activation to BF16 before the
up-projection multiply, preserving the reference operation boundary.

BF16 execution does **not** give bitwise Transformers parity: the final 32-token
comparison has logit RMS error 0.1744 (relative RMS 4.21%), maximum error 1.96875,
and 100% top-token agreement. Some expert selections differ. BF16 rounding,
different GEMM shapes, and routing sensitivity can change subsequent refinement.
Full FP32 checks establish model semantics; broader evaluation is still needed
to quantify task-quality differences for BF16 batched execution.

## Current-block timing

Means of ten synchronized forwards after warmup, BF16 with fused activation:

| Committed prefix | Current 32 tokens | Full prefix + current | Ratio |
| ---: | ---: | ---: | ---: |
| 0 | 43.35 ms | 42.33 ms | 0.98× |
| 256 | 42.67 ms | 151.47 ms | 3.55× |
| 1,024 | 44.19 ms | 384.07 ms | 8.69× |

The full path recomputes all hidden states and projects all logits, as in the
literal reference loop. Multi-block forwards currently dispatch experts
individually; the 32-token CUDA path uses batched expert GEMMs. These ratios
compare complete execution paths, not only the isolated K/V cache optimization.
At zero prefix both paths do the same work; the timing difference is noise.

A separate run keeps individual expert dispatch on both paths: 69.29 ms cached
versus 380.40 ms full, or **5.49×** from prefix reuse without batched dispatch.
Batched experts reduce that cached forward from 69.29 to 44.19 ms, about 1.57×.

Disabling only activation fusion gives current-block means of 43.97, 44.72,
and 45.07 ms respectively. Fusion reduces forward time by approximately 1–5%
in this sample. FP32 activation tensors were a measurable, secondary cost here.
FP32 arithmetic in registers is retained. RMSNorm/attention fusion and avoiding
contiguous prefix K/V copies remain optimization opportunities.

## Generation and HTTP checks

Same chat template and greedy decoder settings, warmed implementations,
sequential model residency. End-to-end generation times exclude model loading.
Cases are in [generation_cases.json](../tests/generation_cases.json).

| Case | Generated tokens | minnow | Transformers | minnow tokens/s | Exact text |
| --- | ---: | ---: | ---: | ---: | --- |
| Arithmetic | 96 | 0.982 s | 2.034 s | 97.8 | Yes |
| Short factual answer | 9 | 0.095 s | 0.123 s | 94.5 | Yes |
| Python function | 28 | 0.322 s | 0.457 s | 86.8 | Yes |
| Longer prompt | 128 | 2.621 s | 8.509 s | 48.8 | No |

The longer response differs in wording and formatting. Arithmetic and longer
responses hit their requested token limits; this small sample does not establish
equal answer quality. Different refinement counts also affect generation timing.
The arithmetic response used 22 forwards and processed 704 token positions.

HTTP checks passed for health/model listing, chat and raw completions, token
usage/template agreement, and rejection of invalid model, streaming, context
overflow, empty prompts, unknown fields, and mixed prompt/messages requests.
An explicit cross-language lock check confirms that a Python-held model lease
rejects a Rust load before any weights are allocated.
CPU and CUDA-feature test suites and Clippy pass; the two explicit CUDA kernel
tests also pass. Validation servers are stopped after the checks.
