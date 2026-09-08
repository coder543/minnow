# minnow implementation notes

Minnow implements LLaDA2.2-mini and LLaDA2.2-flash in Rust with Candle and
custom CUDA kernels. Supported weight formats are BF16, INT8, and NVFP4, with
mixed precision by expert layer/projection. Python is used only for independent
reference validation.

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
- Keep one resident weight set per instance across its requests. A bounded LRU
  pool retains independent conversation K/V slots; reuse requires exact complete
  token blocks.

## CUDA execution

Each instance streams its checkpoint into one resident weight set, shared across
its requests. Separate server processes are independent. See
[memory ownership](memory.md) and [validation](validation.md).

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

The optional BF16 GPU routing path computes the block-capacity selection and top eight
experts, produces GEMM pointer arrays on the device, and mixes their outputs.
It uses 48 expert slots, padding unused slots with references to an already active
expert; no weight copy is made. It is disabled by default: removing CPU round
trips did not offset the extra GEMM work on GB10. See the measured ablations in
[decode performance](decode-performance.md).

Transformer prefill batches and attention query tiles are independent. A query
tile sees only committed keys and keys up to its own final block. This avoids
computing attention to later tiles and bounds score storage without repeatedly
visiting transformer weights. With `--materialized-attention`, the block-softmax
kernel keeps scaling, masking,
and FP32 reductions in registers, for key lengths up to 131,072. It preserves the
BF16 score/probability rounding points, with a different FP32 reduction tree.
Beyond 8,192 keys, a bounded-register kernel makes multiple passes over BF16
scores. This is tiled explicit attention,
not FlashAttention: QK scores and probabilities are still materialized in BF16.

The default CUDA attention path uses a specialized FlashAttention/CUTLASS
forward kernel with 64-query/64-key tiles and a 32-token block-causal mask. It
reads the strided K/V cache directly, writes token-major output, and keeps score
and probability tiles on chip. BF16 score/scaled-score rounding is retained,
but online softmax changes probability rounding and accumulation order. It is
the default on supported shapes; `--materialized-attention` provides the prior
comparison path. See [measurements](nvfp4-optimizations.md).

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
BF16 tensor-core operands for INT8 or native block-scaled FP4 operands for
NVFP4, with FP32 accumulation. INT8 dequantizes in registers; NVFP4 quantizes
activation rows dynamically and shares the gate/up input. Optional lossless fragment packing makes weight loads coalesced.
NVFP4 gate and up projections share a launch with SiLU/multiply while retaining
the original BF16 rounding points. The single-block NVFP4 path computes routing
and compact descriptors in one GPU kernel. It sorts the 256 assignments into
expert order, issues at most two 16-row tiles per active expert, and mixes the
compact output in the original FP32 order. Routing, descriptors, and mixture
weights never need a host round trip. `--host-routing` selects the comparison
path. Larger batches and unsupported/mixed projection combinations use host
routing with grouped GEMMs; no weights are duplicated.
Mixed layer/projection precision shares the same expert gather/mix pipeline.
Shared experts, attention, routers, embeddings, and the head retain source precision.

## Sources

- [Checkpoint and model card](https://huggingface.co/inclusionAI/LLaDA2.2-mini)
- [Reference model code](https://huggingface.co/inclusionAI/LLaDA2.2-mini/blob/main/modeling_llada2_moe.py)
- [Candle](https://github.com/huggingface/candle)
- [cuBLAS grouped GEMM, including BF16 and pointer-array requirements](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublasgemmgroupedbatchedex)
