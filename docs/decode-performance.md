# Decode optimization measurements

Native BF16 LLaDA2.2-mini on NVIDIA GB10, Candle 0.11.0 and CUDA 13.0.3.
On the four existing smoke cases, decode is about 20–24% faster than the fresh
prefill-optimized baseline. These cases do not establish sustained decode throughput
or a short-versus-long response speed relationship. See the
[matched-prompt length sweep](decode-length-sweep.md) for longer generations.
The default path retains CPU routing. [Raw measurements and source hashes](decode-performance.json).

## Useful generation

Each case has one warmup and five measured complete generations. Throughput counts
only final returned model tokens, including role/end markers, divided by generation
time. That time includes prefill and every refinement/commit forward. Model
loading, prompt tokenization, and final text detokenization are excluded.
Both columns use the same native BF16 checkpoint, prompts, token limits, seed,
thresholds, and decode schedule. Profiler runs are separate from these timings.

| Case | Prompt / output tokens | Before useful tok/s | After useful tok/s | Speedup | Denoise / total forwards |
| --- | ---: | ---: | ---: | ---: | ---: |
| arithmetic | 25 / 96 | 106.9 | 128.7 | 1.20× | 19 / 22 |
| factual | 21 / 9 | 102.2 | 126.3 | 1.24× | 2 / 2 |
| code | 29 / 28 | 95.9 | 115.7 | 1.21× | 6 / 7 |
| longer_prompt | 72 / 128 | 54.0 | 65.2 | 1.21× | 52 / 57 |

All output tokens and all forward counts are unchanged in every measured run.
The factual case contains seven text tokens and two special end markers. Its
nine output tokens share a 32-token block with 21 prompt tokens and require only
two denoising passes. The 128-token `longer_prompt` case requires 52 passes over
five blocks. Their different rates reflect different convergence work; comparing
them does not isolate response length. The factual text-only rate is 98.2 tok/s.
The measured gains are arithmetic: 20.3%, factual: 23.6%, code: 20.6%, longer_prompt: 20.9%. These are four smoke cases, not a broad quality
benchmark. The fixed arithmetic and longer-response limits truncate some outputs.
The earlier Transformers comparison remains in [the prefill report](throughput-comparison.md);
it was not rebenchmarked in this pass. HTTP checks still match three of the four
saved SDPA responses exactly, with different wording in the longer response.

## Changes

- Fused RMSNorm retains FP32 arithmetic and the cast to BF16 before the learned
  scale, while removing the separate intermediate tensors and launches.
- Fused Q/K normalization, partial RoPE, and head layout conversion retain the
  existing FP32 reduction tree and BF16 rounding points. RoPE tables are reused
  while refining the same block, with one table pair retained per request.
- Greedy selection uses tiled maximum/sum reductions rather than materializing
  vocabulary-wide softmax probabilities. Only token IDs and confidences transfer
  to the CPU. The FP32 sum tree changes; the CUDA test checks it against FP64.
  Ties now choose the lowest vocabulary index, matching PyTorch's argmax.
- Added repeated generation measurements and a cached-only forward benchmark.

For the arithmetic profile, GPU kernel launches fell from **35,845
to 11,796**. Greedy selection fell from about **3.46 ms
to 27.3 μs per refinement**, including the old argmax and gather kernels.
The new norm and QKV kernels are bitwise identical to the previous explicit
operators in their numerical tests.

## What limits further speed

Matrix multiplication now accounts for **96.3% of GPU kernel time**
in the arithmetic profile. The GPU is executing kernels for about
93.0% of the captured GPU span. This is not primarily
a launch-latency problem anymore.

The 32-token block touches an average of **45.2 distinct experts per MoE
layer** in this profile, although each token selects only eight. All those
distinct expert weights must be read for a refinement. That is why the per-token
active-parameter count understates this workload's weight traffic when comparing
it with an autoregressive model. The diffusion advantage depends on how many
useful tokens each refinement commits; it is smaller in the longer case, which
needs 52 denoising forwards for 128 returned tokens.

Alternative cuBLAS grouping and prototype WMMA tiles did not improve large expert
batches. A bounded synthetic test of 24–48 BF16 experts reached roughly 200 GB/s
when dividing unique weight bytes by execution time. That was an effective rate,
not a hardware measurement of DRAM traffic; the GEMM time share alone does not
prove bandwidth saturation. The [follow-up measurements](decode-length-sweep.md)
check L2/system-memory counters, longer responses, and redundant block-commit
forwards. Quantization and CUDA graphs are not implemented here.

GPU routing was implemented and checked bitwise. It removes CPU routing transfers
but uses fixed-capacity cuBLAS batches, whose unused slots still execute GEMMs.
With the other fusions enabled, CPU routing measured about 0–6% faster across
these cases. Consequently GPU routing is an opt-in diagnostic (`--device-routing`),
and the default runs only the active expert batches. Neither path copies weights.

## Prefix length and prefill

The cached-only diagnostic repeatedly evaluates the same masked 32-token block,
including the vocabulary projection, with ten timed forwards after a warmup.
It excludes prefill and greedy selection; it measures forward latency, not useful
output throughput. Both columns are from the same build, with norm/QKV fusions
disabled for the first column. Both retain the RoPE table cache and CPU routing.

| Committed prefix tokens | Unfused norm/QKV, ms | Fused norm/QKV, ms |
| ---: | ---: | ---: |
| 0 | 18.58 | 14.43 |
| 256 | 29.14 | 25.09 |
| 1,024 | 34.76 | 30.55 |
| 4,096 | 35.48 | 31.17 |
| 8,160 | 35.95 | 31.58 |

The final row has 8,192 total tokens including the current block. The zero-prefix
all-mask block routes to fewer experts and is not representative of complete
generation. Longer contexts, including the full 128K limit, remain unbenchmarked.

The same fusions also improved prefill. These measurements use the actual serving
prefill path, two warmups and five timed runs per length, 4,096-token transformer
batches and 1,024-query attention tiles, with the same input as the prior pass.

| Prompt tokens | Prior prefill pass, tok/s | Current, tok/s |
| ---: | ---: | ---: |
| 512 | 3,868 | 4,185 |
| 2,048 | 6,135 | 7,200 |
| 4,096 | 6,222 | 7,314 |
| 8,192 | 5,307 | 6,253 |

## Correctness and memory

All saved tensors from the trained 512-token BF16 full forward, and from the
480-token prefill plus 32-token cached forward, are **bitwise identical to the
previous optimized implementation**. The experimental GPU routing result is
also identical on that cached trace. Against eager Transformers, the final
32-token block's existing BF16 discrepancy is unchanged: logit RMS error 0.18884,
maximum error 1.25, and 30/32 top predictions matching.

The trained 96-token FP32 diagnostic still passes every tensor at
`atol=1e-3, rtol=2e-5`, including exact router IDs. Maximum logit error is
0.000326633; cached-versus-full maximum error is
0.000274181. All 12 standard tests and 10 CUDA tests pass;
CPU and CUDA clippy checks pass, and HTTP smoke checks pass.

Generation peaked at **31.80 GiB of system-memory growth**. The largest BF16
validation/benchmark peak was **33.44 GiB**, including one 30.28 GiB checkpoint
weight set. The separate FP32 diagnostic peaked at
62.86 GiB with one 60.56 GiB weight set.
All large runs were sequential under the memory guard, with **zero new swap-out**.
Checkpoint loading still uses bounded direct reads from local NVMe and no mmap.

## Reproduce

```sh
cargo build --release --features cuda
cargo test --release --features cuda -- --include-ignored --test-threads=1

python3 scripts/memory_guard.py --report artifacts/decode-current-memory.json \
  --max-growth-gib 40 --reserve-gib 40 -- \
  target/release/minnow --model models/mini-bf16.mnw decode-bench --cases tests/generation_cases.json --iterations 5

python3 scripts/memory_guard.py --report artifacts/decode-prefix-memory.json \
  --max-growth-gib 40 --reserve-gib 40 -- \
  target/release/minnow --model models/mini-bf16.mnw bench --input artifacts/decode-validation/forward-input.json \
  --cached-only --prefix-blocks 0,8,32,128,255 --iterations 10
```

The prefix diagnostic input is the existing 4,320-token documentation seed with
32 mask tokens appended. `bench` cycles it for longer prefixes, replacing masks
in the committed prefix with spaces. Add `--unfused-norm --unfused-qkv` for the
same-build forward ablation, or `--device-routing` for the optional route path.
For Nsight Systems, add `--profile` to `decode-bench` and capture with
`--capture-range=cudaProfilerApi --capture-range-end=stop`.
