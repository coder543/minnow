# Custom CUDA kernel target comparison

This compares `MINNOW_CUDA_ARCH=compute_80` and `compute_121` on GB10 using
CUDA 13.0 (nvcc V13.0.88), native BF16 model weights, and the same runtime source.
The variable controls only `src/kernels.cu` and its includes. Candle's kernels
remain built for `sm_121f`; expert GEMMs continue to use the same cuBLAS library.
No quantization, sampler, or cache algorithm changes are involved.

## Results

No material throughput improvement was measured. The newer target is 0.02–0.40%
lower across the aggregate prefill and useful-decode rates, smaller than the
observed run-order drift. The portable custom-kernel default remains `compute_80`;
`compute_121` is available as an explicit build option.

| Workload | `compute_80` tok/s | `compute_121` tok/s | Change |
| --- | ---: | ---: | ---: |
| Prefill 2,048 | 7,214.30 | 7,186.71 | -0.38% |
| Prefill 4,096 | 7,281.30 | 7,252.44 | -0.40% |
| Decode lhc (293 useful tokens) | 42.71 | 42.64 | -0.17% |
| Decode react_typescript (895 useful tokens) | 113.01 | 112.94 | -0.07% |
| Decode fellowship (750 useful tokens) | 40.04 | 40.03 | -0.02% |

| Fixed 32-token forward: prefix | `compute_80` ms | `compute_121` ms |
| --- | ---: | ---: |
| 0 | 33.303 | 33.335 |
| 1,024 | 33.702 | 33.755 |
| 4,096 | 35.472 | 35.589 |

All 36 measured generations have matching token IDs, finish reasons, and work
counters across the builds. The outputs require 203, 238, and 563 refinements
for LHC, React, and Fellowship respectively. Warmup equality also passes.

The guard recorded 32.67 GiB peak growth above the unloaded
system baseline and no new swap-out. Every benchmark process released its model
before the next phase.

For context, the baseline 4K prefill process means changed from 7,360 to 7,204
tok/s between the first and last rounds; the newer target measured 7,264 and
7,241 in the middle rounds. GPU temperatures rose over the run, while the sampled
SM clocks stayed around 2.4 GHz. The result supports no material gain from this
flag for the current kernels, rather than a claim that `compute_121` is inherently
slower or that architecture-specific kernel redesign cannot help.

[Sample times, per-process rates, binary/source hashes, and memory report](cuda-architecture.json).

## Numerical prerequisite

Simply changing the target initially failed the existing bitwise QKV/RoPE test
at the first output element: BF16 bits 48819 versus 48820. The newer target
emitted BF16 multiply/add instructions that the final compiler could contract
into FMA, dropping an intermediate rounding point. Offline assembly for `sm_121`
confirmed a fused `HFMA2.BF16_V2` in the RoPE expression. `--fmad=false` on the
PTX-generation command alone did not preserve that boundary through the driver
JIT.

The RoPE expression now uses `__hmul_rn` and `__hadd_rn`. NVIDIA documents these
as preventing multiply/add contraction; intermediate products remain in
registers and round to BF16 before addition. See the
[NVIDIA BF16 arithmetic reference](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html).
Both benchmark binaries include this change and pass all 33 tests, including
bitwise QKV/RoPE and all-finite-BF16 SiLU checks. The failed build was not used for
the throughput comparison.

## Method

The order is A/B/B/A, with A=`compute_80`, B=`compute_121`. Each phase starts a
fresh process, loads one checkpoint, warms up, measures, and exits before the
next process starts. All large runs are under the system memory guard; no
concurrent full model, profiler, or compilation runs alongside the timings.

- Prefill: 2,048 and 4,096 tokens, two warmups and five samples per length per
  process, giving ten measured samples per target and length. These execute the
  serving prompt-to-K/V path without vocabulary projection. Transformer batches
  are 4,096 tokens and attention query tiles are 1,024 tokens.
- Fixed work: a 32-token forward at prefix lengths 0, 1,024, and 4,096, one warmup
  and 30 timed forwards per prefix per process. This includes vocabulary
  projection but excludes prediction/decoding. The current token block is fixed.
- Useful decode: the LHC, React TypeScript, and Fellowship prompts in
  `tests/decode_natural_cases.json`, with the normal 2,048-token cap and EOS enabled.
  Each case has one warmup and three samples per process, giving six measured
  responses per target and prompt. Rates count final non-special token IDs and
  divide by decode time, including refinement and commit work but excluding
  prefill. Outputs and work counters are compared across targets.

Loading and CLI serialization are excluded from the timed sections. Rates use
summed token counts divided by summed times. Small differences should be viewed
alongside the per-process rates and sample times in the JSON report; A/B/B/A
reduces order bias but does not establish statistical significance.

## Reproduction

Build and test each target before copying its binary, using the same source:

```sh
mkdir -p artifacts
MINNOW_CUDA_ARCH=compute_80 cargo test --release --features cuda -- --include-ignored
MINNOW_CUDA_ARCH=compute_80 cargo build --release --features cuda
cp target/release/minnow artifacts/minnow-compute80
MINNOW_CUDA_ARCH=compute_121 cargo test --release --features cuda -- --include-ignored
MINNOW_CUDA_ARCH=compute_121 cargo build --release --features cuda
cp target/release/minnow artifacts/minnow-compute121
```

Unload the serving model before running:

```sh
python3 scripts/memory_guard.py \
  --report artifacts/arch-comparison/memory.json --max-growth-gib 40 --reserve-gib 40 -- \
  python3 scripts/bench_cuda_arch.py \
  --a artifacts/minnow-compute80 --b artifacts/minnow-compute121 \
  --input tests/prefill_benchmark_ids.json
```

The harness writes full outputs to `artifacts/arch-comparison/raw.json`, individual
phase JSON/logs, and a compact `summary.json`. The prefill token IDs are fixed
across both targets and recycled only when the requested length exceeds the seed.
