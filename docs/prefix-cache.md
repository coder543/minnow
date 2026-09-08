# Conversation slots and the long-prefill slowdown

Measured on GB10 with the mini BF16 checkpoint, 4,096-token transformer prefill
batches, and attention query tiles capped at 1,024 tokens. The checkpoint and
weight precision are unchanged. See [numeric measurements](prefix-cache.json)
and [API controls](api.md#conversation-prefix-slots).

## Why the captured rates differed

The attention path had an 8,192-key cutoff. Above it, fused block softmax fell
back to separate FP32 scaling/softmax intermediates and a CPU-built block mask
uploaded to the GPU. The two similarly sized captures had 7,904 and 8,448
complete prompt tokens, placing them on opposite sides of that cutoff.

In the original 18,624-token capture, prefill reached 8,192 tokens after 1.320 s.
The remaining 10,432 tokens took another 18.754 s. Replaying the same four
captured prompts reproduced the slowdown. The replacement uses a bounded-register
softmax kernel for long rows, computing the mask in the kernel and rereading BF16
scores instead of allocating FP32 intermediates. It supports 131,072 keys and
retains the existing score/probability rounding points.

| Complete prompt tokens | Before, tok/s | After, tok/s | Speedup |
| ---: | ---: | ---: | ---: |
| 7,904 | 6,141 | 6,328 | 1.03× |
| 8,448 | 5,186 | 5,873 | 1.13× |
| 10,240 | 2,425 | 5,558 | 2.29× |
| 18,624 | 907 | 4,023 | 4.44× |

These are **cold** requests, with one output token solely to complete the HTTP
request. The denominator is synchronized prefill time, not generation time.
There are two measurements per length except 7,904: its first sample after each
server load is excluded for dispatch warm-up (4,880 before / 4,895 after).
Captured prompt bodies and token IDs remain in ignored local artifacts.

The residual decline is expected for this explicit attention implementation:
longer prefixes require more QK/PV work, and the workspace cap reduces query tile
size. The kernel still materializes BF16 attention scores and probabilities;
this change is not FlashAttention. Batch size, input-dependent expert routing,
and first-use dispatch costs also affect throughput. Full 128K generation and
prefill throughput have not been benchmarked.

## Prefix reuse

Four slots are enabled by default, sharing a 5 GiB K/V capacity budget for mini
BF16. Longest exact block-prefix matching chooses the source. A sufficiently
similar request truncates that slot in place; a divergent request copies the
shared complete blocks to an empty or LRU slot when the budget permits. The
similarity threshold affects retention only: nonmatching token blocks are always
recomputed. Each server instance has one inference worker and one resident
weight set.

Replaying the captured conversation sequence gave:

| Request's complete prompt tokens | Reused | Newly prefilled | New prefill time | Cache setup/copy time |
| ---: | ---: | ---: | ---: | ---: |
| 8,448 following 7,904 | 7,904 | 544 | 197.4 ms | 7.1 ms |
| 18,624 following 10,240 | 10,240 | 8,384 | 2,866.5 ms | 18.9 ms |
| Exact repeat of 18,624 | 18,624 | 0 | 0.02 ms | 0.05 ms |

These exclude tokenization, queuing, and subsequent decoding. Small uncached
suffixes can report a lower prompt tok/s despite much less latency: their rate
counts only the newly computed tokens. A full cache hit reports zero new prompt
tokens and zero prompt tok/s, with the reuse exposed separately in `cache_n` and
OpenAI usage. It skips prefill progress events and emits `prefill_complete`
before refinement events, clearing the UI's preparing state.

## Validation and numerical limits

Rust tests pass, including CPU fixture checks against the reference and CUDA
kernel tests. New tests cover independent fork storage, LRU selection,
truncation inside a divergent block, K/V buffer growth, memory-budget eviction,
cancellation, model identity, and per-request work counters. Long softmax tests
cover rows just above 8K and at 131,072 keys, with at most one BF16 rounding step
of error versus explicit FP32 softmax and exactly zero probability for masked
future tokens.

`scripts/check_prefix_cache.py` passes against the real server: four conversations
coexist, forks preserve the source, buffer growth preserves output, cold requests
bypass reuse, and full/partial hits have correct SSE progress and usage.
`scripts/check_compatibility.py` also passes, including Chat Completions, streamed
tool calls, tool results, stop strings, long prefill, and disconnection handling.

Warm/cold output is **not guaranteed bitwise identical when prefill batches
change**. Two 64-token continuations at 8,473 and 18,635 prompt tokens changed
wording when cached prefixes had been built in different batches. Consecutive
warm repeats matched exactly. This warranted checking cache correctness rather
than assuming the difference was harmless.

The full-model CUDA storage regression now checks every committed K/V element
and subsequent logits after copying to a different capacity, overwriting the
fork independently, truncating and recomputing its last block, growing a buffer,
and discarding uncommitted refinement scratch. These checks are **exact**, using
the same computation shapes, and pass on the 8,448-token captured prefix. This
isolates storage operations from batch-dependent floating-point computation.

`trace_prefill_boundaries` isolates the other variable: both executions prefill
4,096 identical tokens, then evaluate either 4,096 or 3,808 more tokens with the
same cache capacity. It compares the 3,808 common rows. BF16 embeddings, layer 0,
and layer 1 attention agree exactly. The first difference is the FP32 router
matrix multiplication in layer 1 (maximum 2.38e-6); its expert selections still
agree. The layer's residual output differs by at most 0.000488, and layer 2 first
selects different expert sets for 7 of 3,808 rows. The differences grow through
later layers. Disallowing reduced-precision cuBLAS reductions produces exactly
the same trace, so that setting is not a remedy here.

The full FP32 control now completes under the revised memory guard, holding one
60.6 GiB weight set. Its first differences are about 3e-6 in layer 0 attention,
and expert sets first differ in layer 11 for one row. For first-refinement logits,
`check_prefix_numerics` measures:

| Weight/activation precision | RMS difference | Maximum difference | Matching top predictions |
| --- | ---: | ---: | ---: |
| BF16 | 0.47656 | 5.64063 | 29/32 |
| FP32 | 0.01166 | 0.35759 | 32/32 |

These results identify batch-dependent arithmetic amplified by discrete MoE
routing in this case, rather than corrupted K/V. They do not establish equal
answer quality or universal batch invariance. Bitwise batch invariance would
require controlling the arithmetic across GEMM and attention shapes, including
expert batches; changing storage precision or one cuBLAS flag is insufficient.
NVIDIA documents mixed-precision reduction controls in its
[cuBLAS guide](https://docs.nvidia.com/cuda/cublas/index.html#cublasmath-t).

The FP32 diagnostic peaked at 66.54 GiB of memory growth (the more detailed
activation trace used 70.03 GiB). The BF16 cache-storage regression used 32.39 GiB.
All completed with zero new swap-out.

The successful BF16 cold replay peaked at 34.31 GiB of system-memory growth,
including model loading, and produced no new swap-out. Slot growth replaces one
layer at a time; fork copies are included in the aggregate capacity budget.
Only one complete model weight set is resident, and loading uses direct reads
from the local SSD without checkpoint mmap.

```sh
cargo test --release --features cuda -- --include-ignored --test-threads=1
python3 scripts/memory_guard.py --report artifacts/cache-memory.json \
  --max-growth-gib 12 --reserve-gib 40 -- \
  python3 scripts/check_prefix_cache.py --url http://127.0.0.1:8080
```

The HTTP check assumes an already resident server; use a 40 GiB growth ceiling
if the check also triggers loading through a proxy. Run full-model diagnostics sequentially with serving
stopped and under the corresponding memory guard.
