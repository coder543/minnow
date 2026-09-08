# Sustained decode and expert memory traffic

Native BF16 LLaDA2.2-mini on GB10. These measurements separate response length,
convergence work, and final text tokens. [Raw data, output IDs, source hashes,
and memory reports](decode-length-sweep.json).

## Natural responses

The prompts are recorded in [decode_natural_cases.json](../tests/decode_natural_cases.json).
Each has a 2,048-token maximum and normal EOS stopping. All three stopped naturally
below that maximum. Each column uses one warmup and three measured complete
generations, with temperature 0, threshold 0.5, editing threshold 0, steps 32,
max post steps 16, and seed 42. Times include prefill, refinement, and any commit
refresh. Loading, prompt tokenization, and final detokenization are excluded.

Throughput counts final non-special token IDs. Each response also contains two
special end/role markers, excluded from these rates. Refinement passes include
all evaluations in the decoder, including its convergence check. They are per
whole 32-token block, not per returned token.

| Prompt | Text tokens | Blocks | Refinement passes | Passes/block | Before text tok/s | After text tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| What is the LHC? | 293 | 10 | 203 | 20.30 | 40.65 | 42.35 |
| React TypeScript example | 895 | 29 | 238 | 8.21 | 100.64 | 111.79 |
| Fellowship characters and backstories | 750 | 25 | 563 | 22.52 | 38.16 | 39.66 |

The React example requires far fewer refinements per block than the two factual
prompts. This is an observation about these exact prompts and settings; it does
not establish a general advantage for code. Outputs and refinement counts are
identical across repetitions and before/after the cache change.

## Same prompt, different output limits

The earlier executable, before conditional commits, generated the same tutorial
and story at 128, 512, and 1,024-token limits. Each point has one warmup and three
timed repetitions. All reached the cap, with no special tokens in their returned
IDs. The shorter outputs are exact prefixes of the longer outputs. Prompts are
in [decode_throughput_cases.json](../tests/decode_throughput_cases.json).

| Prompt | Text tokens | Blocks | Passes/block | Text tok/s |
| --- | ---: | ---: | ---: | ---: |
| tutorial | 128 | 5 | 16.20 | 43.72 |
| tutorial | 512 | 17 | 16.82 | 48.98 |
| tutorial | 1024 | 33 | 15.30 | 54.90 |
| story | 128 | 5 | 23.20 | 31.52 |
| story | 512 | 17 | 27.88 | 31.10 |
| story | 1024 | 33 | 26.61 | 33.64 |

The tutorial gets faster with length in this sweep; the story is roughly flat.
The earlier seven-text-token factual smoke case was not evidence that shorter
responses decode faster. That block also held 21 prompt tokens and two end
markers, and took only two denoising passes. Counting its nine model tokens gave
126.3 tok/s; counting seven text tokens gave 98.2 tok/s. It does not characterize
sustained long-output throughput. The [historical smoke report](decode-performance.md)
now states that limitation explicitly.

`decode-bench` now defaults to this length sweep. It reports model tokens, text
tokens, decode-only text throughput excluding prefill, passes per block, and text
tokens per refinement. Its legacy `useful_tokens_per_second` field remains an
explicit alias of model-token throughput for compatibility.

## Removing redundant commit forwards

A successful model forward records the exact tokens for which it wrote scratch
K/V. Previously, generation evaluated every completed non-final block again,
even after a convergence check had already evaluated its final tokens. It now
commits those existing K/V tensors directly when the token IDs match. If the
decoder terminates after changing tokens, it still refreshes the block first.
This preserves model arithmetic, routing, thresholds, and the decode schedule.

| Prompt | Total forwards before | Total forwards after | Reused commits | Throughput gain |
| --- | ---: | ---: | ---: | ---: |
| What is the LHC? | 212 | 203 | 9 | 4.2% |
| React TypeScript example | 266 | 238 | 28 | 11.1% |
| Fellowship characters and backstories | 587 | 564 | 23 | 3.9% |

The LHC and React runs reused every non-final block's staged K/V. Fellowship
reused 23 commits and correctly refreshed one block whose final tokens had
changed. The test fixture also covers a final edit that requires a refresh, checks the exact
forward-count difference, rejects stale/double commits, and compares the next
block's logits to a recomputed full prefix. All 23 tests, including ten CUDA
tests, passed; CPU/CUDA clippy and formatting checks passed. Peak system-memory
growth across these recorded generation/benchmark runs was 32.50 GiB,
with zero new swap-out. Model runs were sequential under the memory guard,
using one resident BF16 weight copy loaded by bounded direct reads from local SSD.
The four existing smoke cases also retain exact output IDs and denoising counts.
The factual case explicitly verifies the corrected count of seven text tokens
versus nine returned model tokens.

## Testing compact expert inputs

An optional `--compact-decode-experts` switch uses the existing prefill grouped
GEMM implementation on only the rows assigned to each expert. In the earlier
profile, the default evaluates 32 rows for each of 45.2 experts, although routing
assigns only 256 token/expert pairs in total. Compacting reduces logical row work
and activation storage; tensor-core kernels can still pad their internal tiles.

These are cached current-block forward latencies, including the vocabulary head,
with one warmup and 20 timed repetitions per prefix. They exclude prefill and
token selection and are not useful output-token rates. Both modes use the same
saved input, BF16 weights, and other execution settings.

| Committed prefix tokens | Default, ms | Assigned rows only, ms |
| --- | ---: | ---: |
| 256 | 25.076 | 25.334 |
| 1024 | 30.671 | 30.559 |
| 4096 | 31.261 | 31.331 |

This measured no useful speedup, within about 1% in either direction, so the
default remains unchanged. Removing those rows does not remove the expert
weight matrices. The switch remains available for further kernel experiments.
All 81 tensors in its trained 480-token-prefix plus 32-token cached trace were
byte-for-byte identical to the saved default trace, including router IDs and logits.

## What is actually being reread?

The earlier arithmetic profile attributed 96.3% of GPU kernel time to matrix
multiplication and averaged 45.2 distinct experts per MoE layer. That time share
alone does not measure bandwidth saturation. Each selected expert has three
different BF16 matrices: gate, up, and down. At roughly 45 experts over 19 layers,
their unique weights total about 5 GiB per refinement, before dense/shared
matrices, attention projections, and the vocabulary head.

To check redundant system-memory reads without loading or snapshotting another
model, a bounded synthetic benchmark used the production pointer-batched GEMM
calls and shapes: 48 selected experts, 32 rows, and gate/up 512×2048 or down
2048×512. Gate/up share one input across experts; down uses separate expert
activations. Nonconstant BF16 data resides in a 512 MiB expert allocation.
Nsight Compute selected the same kernel names and grids as the serving profile.
Default cache flushing was enabled for these isolated counter measurements.

| GEMM | Unique weights + input | L2 fills from system memory | Excess over unique bytes |
| --- | ---: | ---: | ---: |
| Gate/up shape | 100,794,368 B | 100,922,592 B | 0.127% |
| Down shape | 102,236,160 B | 102,401,472 B | 0.162% |

GB10 exposes system-memory L2 fill counters here; `dram__bytes_read` is unavailable.
These are `lts__d_sectors_fill_sysmem.sum` multiplied by 32 bytes per sector,
as defined in the [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide).
The additional bytes can include pointer arrays and other kernel data. L2 read
requests themselves total 185.0/189.5 MB, so some repeated requests are served
from cache. This experiment finds almost one system-memory read of the unique
weights and inputs, not a large redundant weight-read multiplier. It does not
measure the entire transformer or prove that every production GEMM is optimal.
The source and counter CSVs are in `artifacts/profile-expert-traffic.cu` and
`artifacts/expert-traffic-{gate,down}.csv`.

## FP32 intermediates and D2R

The expert GEMMs already pass resident BF16 A/B/C buffers to cuBLAS with
`CUBLAS_COMPUTE_32F`. There is no full FP32 expansion of expert weights in RAM.
BF16 operands and FP32 computation are a supported
[cuBLAS combination](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex).
Fused normalization, SiLU/multiply, QKV preparation, softmax, and expert mixing
perform FP32 work in registers or on-chip shared memory and store BF16 outputs.
Since these weights are not quantized, there is no quantized-weight
dequantization step to call D2R. The desired property—avoiding an expanded
weight tensor in memory—is already present. cuBLAS may use accumulator scratch;
this is not a claim that every internal FP32 value is always register-resident.

Some FP32 activation buffers remain: router inputs/logits, and the vocabulary
logits after the BF16 head GEMM. At 32 tokens the latter occupies about 19.2 MiB.
A BF16-aware prediction kernel could remove that conversion buffer. This is
small compared with the several GiB of unique weights read per refinement.

## Further work without quantizing weights

1. Skip redundant whole-model work when it is provably unnecessary, as the
   conditional commit now does. This removes weight reads and compute together.
2. Investigate specialized expert tiling if it improves on the measured grouped
   path. The current batches evaluate about 5.6 times as many logical rows as
   routing assigns in the profiled case. The simple compaction experiment above
   did not improve latency; it still requires the same expert weights.
3. Fuse gate/up with activation and eventually down/mixing where tiling permits.
   This can save intermediate traffic and launches while preserving BF16 rounding
   points. Gate and up remain different matrices; fusion must read both.
4. Batch independent requests and group their assignments by expert to amortize
   overlapping weight reads. This targets aggregate throughput and needs latency
   measurements; it does not promise the same gain for one request.

Caching expert outputs just because a token ID did not change is invalid:
bidirectional attention can change that token's hidden state when its neighbors
change. Altering routing, reducing the block's expert capacity, or stopping
refinement early would change the computation and need separate quality checks.
The GPU's L2 is 24 MiB; the weights touched by a refinement cannot all persist
there between passes.

## Reproduction

```sh
cargo build --release --features cuda
python3 scripts/memory_guard.py --report artifacts/natural-memory.json \
  --max-growth-gib 40 --reserve-gib 40 -- \
  target/release/minnow decode-bench \
  --cases tests/decode_natural_cases.json --iterations 3

python3 scripts/memory_guard.py --report artifacts/length-memory.json \
  --max-growth-gib 40 --reserve-gib 40 -- \
  target/release/minnow decode-bench --iterations 3
```

These commands exercise the current conditional-commit implementation. The
historical length table above used the saved earlier executable; both source
hash sets and its binary hash are preserved in the raw report.
