# minnow implementation notes

Targets: LLaDA2.2-mini and LLaDA2.2-flash, with BF16 or mixed quantized experts,
on NVIDIA GB10. Rust runtime
with Candle; Python is used only for independent reference validation. The model
checkout at `~/hf/inclusionAI/LLaDA2.2-mini` is read-only. The default working
copy is `~/models/hf/inclusionAI/LLaDA2.2-mini`, on the local SSD.

## Execution invariants

- Attention is causal **between blocks**, bidirectional **within each block**.
  Prefill uses the same block mask, not a token-causal or fully bidirectional mask.
- Routing first takes the maximum biased sigmoid score across 32 tokens, selects
  48 allowed experts, then selects 8 per token. Expert bias affects selection only.
  Weights are normalized unbiased sigmoid scores times 2.5, computed in FP32.
- Only complete, committed blocks can enter the persistent KV cache. A partial
  prompt block must be refined together with its generated suffix: its prompt
  hidden states and routing are not frozen yet.
- A denoising forward's K/V describes its **input tokens**. Commit that K/V directly
  when those tokens exactly match the final block. If the final edit changed the
  tokens, refresh before committing. Never commit stale pre-edit K/V; a successful
  forward records its exact token IDs, and commit checks them and clears the record.
- DELETE and SPLIT operate within the fixed block; track original versus newly
  inserted masks and preserve the prompt. Termination and edit suppression follow
  the checkpoint's joint decoder. A hard step limit must report non-convergence.
- Logits are for the same positions, with no autoregressive shift.
- Keep one resident weight set across requests. A bounded LRU pool retains
  independent conversation K/V slots; reuse requires exact complete token blocks.

## Work and verification

1. Implement model/config/checkpoint loading, block attention, partial RoPE,
   FP32 routing, dense/shared/routed SwiGLU, and explicit cache commit.
2. Validate against fixtures produced by the checkpoint's actual Python classes:
   small FP32 model, intermediate outputs, cache equivalence, decoding edge cases.
3. Load one real BF16 checkpoint on CUDA, compare stored reference logits and generated
   text, measure latency and processed-token counts.
4. Provide CLI generation and a bounded HTTP inference worker, health/model
   endpoints, and text chat/completion endpoints with the checkpoint template.
5. Profile and improve material bottlenecks; record reproducible measurements.

Future work includes FlashAttention, native low-bit activation/tensor-core paths,
RTX 3090 hardware validation, and Responses API/session resumption.

See [memory ownership](memory.md): no file-backed mmap, no complete host/CUDA
weight duplication, no concurrent full-model validation. Use direct I/O and the
one-layer reference script. The full FP32 Transformers loading path was removed
after it exhausted system memory; do not reintroduce it.

Current CUDA execution uses pointer-batched expert GEMMs for a single block and
variable-size grouped GEMMs for prefill, with BF16 weights and FP32 accumulation.
Grouped execution gathers assigned token rows once and references the existing
packed expert weights; it neither pads token rows nor copies weights. Expert
gathering/weighting/reduction and SiLU/multiply are fused, preserving their BF16
rounding and reduction order.

RMSNorm now fuses the FP32 cast, square/reduction, normalization, BF16 cast, and
learned scale. The mini/flash QKV kernel also combines head normalization, partial RoPE,
and the transpose to head-major storage. Both retain the original reduction tree
and BF16 rounding points. Only one pair of RoPE tables is retained per request,
replaced when the current block's position or length changes.

Greedy prediction uses two tiled reductions over the vocabulary, without writing
a full softmax tensor. Its FP32 sum tree differs from the eager softmax, and is
checked against an FP64 oracle. Tied maxima choose the smallest vocabulary index,
matching PyTorch; Candle's CUDA argmax can choose a different tied index.
Stochastic sampling still uses the CPU path and Rust RNG.

An optional GPU routing path computes the block-capacity selection and top eight
experts, produces GEMM pointer arrays on the device, and mixes their outputs.
It uses 48 expert slots, padding unused slots with references to an already active
expert; no weight copy is made. It is disabled by default: removing CPU round
trips did not offset the extra GEMM work on GB10. See the measured ablations in
[decode performance](decode-performance.md).

Transformer prefill batches and attention query tiles are independent. A query
tile sees only committed keys and keys up to its own final block. This avoids
computing attention to later tiles and bounds score storage without repeatedly
visiting transformer weights. The block-softmax kernel keeps scaling, masking,
and FP32 reductions in registers, for key lengths up to 131,072. It preserves the
BF16 score/probability rounding points, with a different FP32 reduction tree.
Beyond 8,192 keys, a bounded-register kernel makes multiple passes over BF16
scores. This is tiled explicit attention,
not FlashAttention: QK scores and probabilities are still materialized in BF16.

The BF16 path can differ from eager Transformers because GEMM
shapes and accumulation order change; full FP32 reference comparisons and small
generation checks are recorded in [results](results.md).

Continuous batching uses bounded host decoder threads that yield transformer
segments to one GPU worker. Dense/MoE/head work concatenates token rows while
attention and RoPE operate independently per sequence. Only complete blocks can
be combined, so block-level expert capacity never spans two conversations.
Admission reserves K/V before launching a decoder; success returns the prefix,
failure discards it and releases its reservation. The CUDA allocator retains a
configurable bounded amount of scratch across the decoder's synchronizations.

The [self-contained format](model-format.md) stores aligned original or quantized
payloads and a checksummed MessagePack manifest. Quantized expert GEMMs use
BF16 tensor-core operands and FP32 accumulation, with code/scale dequantization
in registers. Optional lossless fragment packing makes weight loads coalesced.
Mixed layer/projection precision shares the same expert gather/mix pipeline.
Shared experts, attention, routers, embeddings, and the head retain source precision.

## Sources

- [Checkpoint and model card](https://huggingface.co/inclusionAI/LLaDA2.2-mini)
- [Reference model code](https://huggingface.co/inclusionAI/LLaDA2.2-mini/blob/main/modeling_llada2_moe.py)
- [Candle](https://github.com/huggingface/candle)
- [cuBLAS grouped GEMM, including BF16 and pointer-array requirements](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublasgemmgroupedbatchedex)
