# DGX Spark performance

Fresh LLaDA2.2-mini measurements on NVIDIA DGX Spark (GB10, SM121), CUDA 13,
September 2026. Minnow runtime commit `c2496fc`; Transformers 5.2.0 with PyTorch
2.10.0+cu130. [Measurement data](spark-performance.json) contains individual
timings, output hashes, token counts, refinement counts, and source identities.

| Implementation / expert execution | 4,096-token prefill | LHC decode | React TypeScript decode |
| --- | ---: | ---: | ---: |
| Transformers BF16 / SDPA | 5,808 | 12.7 | 39.5 |
| Minnow BF16 | 10,000 | 48.8 | 131.6 |
| Minnow INT8 / BF16 activations | 9,552 | 73.1 | 232.5 |
| Minnow INT4 / BF16 activations | 10,567 | 102.2 | 273.1 |
| Minnow INT4 / INT8 activations | 12,919 | 105.6 | 272.0 |
| Minnow NVFP4 / native FP4 tensor cores | 15,509 | 95.0 | 279.8 |

## Workload and timing

All runs are sequential with one resident weight copy. Minnow uses BF16 outside
quantized experts, block FlashAttention, 4,096-token prefill chunks, 1,024-query
attention tiles, and a 512 MiB workspace cache. The reference uses its upstream
joint decoder and SDPA attention. INT4/INT8 expert weights use 128-value groups
with FP16 scales. The INT4/INT8-activation row adds `--int8-expert-activations`;
NVFP4 uses native Blackwell FP4 instructions with FP4 weights and activations.

Prefill measures fresh prompt-to-KV construction on the same
[token IDs](../tests/prefill_benchmark_ids.json), excluding loading, tokenization,
the vocabulary head, and generation. Minnow has one warmup and three measured
samples per length; the reference has two warmups and five measured samples.
The data also records minnow's 512-, 2,048-, and 8,192-token prefill results.

Decode uses the exact [comparison cases](../tests/decode_comparison_cases.json):
“What is the LHC?” and “Write a React TypeScript example.” Both implementations
produce identical prompt token IDs (13 and 15 tokens). Neither prompt fills a
32-token block, so there is no separate committed-block prefill to subtract from
reference generation time. Loading and tokenization are excluded. The numerator
counts returned text tokens, excluding special tokens; every refinement is timed.
Rates are total tokens divided by total elapsed time over three measured runs
after one discarded warmup per prompt. Outputs match their warmup within each
execution mode.

Settings are greedy decoding, seed 42, threshold 0.5, editing threshold 0,
max_post_steps 16, and a 512-token output cap. Prefix caching between requests is
disabled. Minnow reuses committed blocks during generation; the reference
recomputes the prefix on each refinement. Different floating-point arithmetic
and quantizations produce different answers and refinement counts. The LHC
responses terminate naturally; every React response reaches the cap:

| Implementation | LHC text tokens | React text tokens |
| --- | ---: | ---: |
| Transformers BF16 / SDPA | 247 | 512 |
| Minnow BF16 | 310 | 512 |
| Minnow INT8 / BF16 activations | 190 | 512 |
| Minnow INT4 / BF16 activations | 191 | 512 |
| Minnow INT4 / INT8 activations | 182 | 512 |
| Minnow NVFP4 / native FP4 tensor cores | 215 | 512 |

These measurements compare observed throughput, not equal task accuracy or
identical work across precisions. The RTX 3090 README table comes from a separate
measurement series with four measured generations and ten prefill samples;
see [its full settings](rtx3090.md).

## Reproduce

Build `cargo build --release --features cuda`. Run full-model measurements
sequentially. The reference needs the Python dependencies described in
[validation](validation.md) and the original safetensors directory.

```sh
python3 scripts/memory_guard.py --reserve-gib 16 --max-growth-gib 50 \
  --report artifacts/spark-bf16-memory.json -- \
  python3 scripts/bench_models.py --model models/llada2.2-mini-bf16.mnw \
  --report artifacts/spark-bf16.json --cases tests/decode_comparison_cases.json \
  --iterations 3 --max-tokens 512 --workspace-cache-mib 512

.venv/bin/python scripts/memory_guard.py --reserve-gib 16 --max-growth-gib 50 \
  --report artifacts/spark-reference-memory.json -- \
  .venv/bin/python scripts/reference_generate.py --model models/LLaDA2.2-mini \
  --attention sdpa --cases tests/decode_comparison_cases.json \
  --prefill-input tests/prefill_benchmark_ids.json --prefill-lengths 4096 \
  --prefill-iterations 5 --generation-iterations 3 --generation-warmups 1 \
  --output artifacts/spark-reference.json
```

For quantized rows, use the corresponding `.mnw` file in the minnow command
and a separate report path. Add `--server-arg=--int8-expert-activations` for the
INT4/INT8-activation row. Guard budgets above are for the 128 GB Spark host.
