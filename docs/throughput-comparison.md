# Prefill and useful generation throughput

This report records the prefill optimization pass. Later decode work and refreshed
prefill checks are in [decode optimization measurements](decode-performance.md).

NVIDIA GB10, native BF16 checkpoint, Candle 0.11.0 versus Transformers 5.2.0 /
PyTorch 2.10.0+cu130. Checkpoints load from local NVMe with direct I/O. Models ran
sequentially under the memory guard. [Raw measurements](throughput-comparison.json).

## Prefill

Two warmups and five timed runs per length, with GPU synchronization at timing
boundaries. Timings cover fresh prompt-to-KV construction, including input/setup,
and exclude model loading, tokenization, vocabulary projection, and decode. Both
implementations receive identical tokens. The benchmark calls the actual serving
`decode::prefill` path. Transformers uses its default SDPA attention backend.

| Prompt tokens | Original minnow tok/s | Optimized minnow tok/s | Transformers SDPA tok/s | Speedup over original |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 981 | 3,868 | 2,546 | 3.94× |
| 2,048 | 959 | 6,135 | 5,280 | 6.40× |
| 4,096 | 877 | 6,222 | 5,739 | 7.09× |
| 8,192 | — | 5,307 | 5,632 | — |

The current default is a 4,096-token transformer batch with 1,024-query attention
tiles. At 4,096 tokens this is about 7.1× faster than the original implementation
and 8% faster than the refreshed SDPA baseline. Throughput peaks around 2,048–4,096
tokens in this sweep; it falls at 8,192 as attention work grows. Setting
`--prefill-chunk-tokens 8192` measured 5,601 tok/s at 8,192 in the tuning sweep,
close to SDPA at that length. This is an observed maximum over these sizes, not
a claim about every context, input distribution, or device.

The original minnow used 128-token chunks. Original eager Transformers rates
were 2,509 / 3,633 / 2,947 tok/s at 512 / 2,048 / 4,096 tokens; those historical
measurements remain in the raw report. The current comparison refreshes SDPA.
A separate CUDA service initialized during the first final timing pass, so the
reported minnow results use a repeat after its GPU activity became idle.

## What changed

- Larger transformer batches amortize expert-weight reads across more tokens.
- Variable-size grouped expert GEMMs operate on assigned token rows, using views
  of the resident weights. No expert padding or extra weight set is created.
- Fused attention scaling, block masking, and softmax remove large FP32 tensors.
  FP32 arithmetic and BF16 score/probability rounding points are retained.
- Fused expert gathering, weighting, and reduction eliminate the large
  `[tokens, top_k, hidden]` intermediates and preserve the existing reduction tree.
- Attention queries are tiled independently of transformer batches, omitting keys
  belonging to later tiles. The KV cache is read through strides without copying
  its committed prefix. This remains explicit tiled attention, not FlashAttention.

Intermediate measurements at 4,096 prompt tokens show where the gains came from:

| Implementation stage | Prefill tok/s |
| --- | ---: |
| Original 128-token chunks, same-session repeat | 884 |
| 2,048-token chunks | 1,748 |
| Add grouped expert GEMMs | 1,928 |
| Add fused attention softmax and KV views | 3,707 |
| Add fused expert mixing | 5,374 |
| Independent attention tiles, 4,096-token batches | 6,222 |

## Correctness and memory

The trained 96-token FP32 diagnostic passes every tensor at `atol=1e-3, rtol=2e-5`,
with all router IDs and top predictions matching. Maximum logit error is
0.000327; cached versus full maximum error is 0.000274. This check also exercises
32-query attention tiles against the full reference forward.

The 512-token BF16 diagnostic contains 480 prompt tokens and a 32-token masked
block. On that final block, logit RMS error versus eager Transformers is 0.217
for the serial/unfused baseline, 0.189 after optimization, and 0.170 after cached
prefill with smaller attention tiles. Each matches 30 of 32 reference top
predictions. BF16 routing and outputs are not bitwise identical; these checks
do not establish broad task-quality equivalence. Full FP32 parity remains the
semantic diagnostic. Raw BF16 comparisons are included in the JSON report.

All 12 standard tests and five CUDA kernel tests pass. CUDA checks include
uneven grouped GEMMs, bitwise expert mixing, exhaustive finite-BF16 SiLU, and
fused softmax within one BF16 rounding step with exactly zero future attention.
CPU/CUDA clippy checks pass. HTTP smoke tests pass; three of four generation
responses match SDPA exactly, with different wording in the longer response.

The final BF16 benchmark peaked at **33.43 GiB of system-memory growth**,
including one 30.28 GiB weight set, runtime allocations, activations, and cache.
Every large validation was guarded and sequential, and none caused new swap-out.
The separate FP32 diagnostic uses one 60.56 GiB weight set under its larger guard.
Weights still use bounded direct reads and never checkpoint mmap.
Attention tiles are bounded to 128 million score elements and transformer batches
to 8,192 tokens. These measurements predate the long-row softmax kernel;
[prefix caching and long prefill](prefix-cache.md) removes the eager fallback
above 8,192 keys. The full 128K path is unbenchmarked.

## Useful generation smoke measurements

Useful throughput counts final returned tokens, including EOS where present,
divided by generation time, including prefill and every refinement/commit.
Loading and tokenization are excluded. These are single warm HTTP smoke runs,
not maximum decode throughput or a quality benchmark. The current minnow column
is refreshed after prefill optimization; SDPA uses the saved matching cases.

| Case | Prompt / output tokens | Original minnow tok/s | Current minnow tok/s | Transformers SDPA tok/s |
| --- | ---: | ---: | ---: | ---: |
| arithmetic | 25 / 96 | 97.8 | 103.9 | 52.3 |
| factual | 21 / 9 | 94.5 | 101.4 | 73.7 |
| code | 29 / 28 | 86.8 | 92.8 | 61.3 |
| longer_prompt | 72 / 128 | 48.8 | 52.6 | 15.5 |

Arithmetic, factual, and code responses match SDPA exactly; the longer response
differs in wording. Responses use the existing fixed token limits, so some are
cut off before completing their explanation. Refinement counts can differ.
The reference joint decoder recomputes the prefix rather than reusing its
standalone prefill cache. Decode tuning remains separate from this prefill work.

## Reproduce

The measured seed has 4,320 tokens of checkpoint documentation/template source,
with reserved edit/mask IDs replaced by a space token. Both tools take the first
requested IDs, cycling the seed for the 8,192-token case. The input hash and
current source hashes are recorded in the JSON report.

```sh
cargo build --release --features cuda
python3 scripts/memory_guard.py --report artifacts/prefill-current-memory.json \
  --max-growth-gib 42 --reserve-gib 40 -- \
  target/release/minnow prefill-bench --input artifacts/prefill-input.json \
  --tokens 512,2048,4096,8192 --iterations 5

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/memory_guard.py \
  --report artifacts/prefill-reference-memory.json --max-growth-gib 42 --reserve-gib 40 -- \
  .venv/bin/python scripts/reference_generate.py --attention sdpa \
  --prefill-input artifacts/prefill-input.json --prefill-only \
  --prefill-lengths 512 2048 4096 8192 --prefill-iterations 5 \
  --output artifacts/prefill-reference.json
```

Add `--chunk-sizes 512,2048,4096,8192` to sweep transformer batches using one
resident model. Change `--attention-chunk-tokens` independently. Use
`--serial-experts`, `--unfused-attention`, or `--unfused-expert-mix` for component
comparisons. `prefill-bench --profile` marks timed iterations for Nsight capture
with `--capture-range=cudaProfilerApi`. Keep all full-checkpoint runs sequential.
