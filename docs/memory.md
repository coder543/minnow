# Memory ownership and loading

The working checkpoint lives at `~/models/hf/inclusionAI/LLaDA2.2-mini` on the
local NVMe SSD. `scripts/copy_checkpoint.py SOURCE DESTINATION` copies it with
bounded direct I/O and verifies SHA-256 hashes before publishing the directory.
The original network checkout remains read-only. The iSCSI workspace stores code
and small validation artifacts, not the working checkpoint.

The runtime holds one complete set of weights. It never constructs a complete
CPU model to copy to CUDA and never maps checkpoint shards. Rust and the Python
reference tools share `/tmp/minnow-model-<uid>.lock`, held for the model lifetime,
to reject overlapping full-checkpoint work. Small test fixtures are exempt.

`src/weights.rs` reads safetensors headers, opens shards with Linux `O_DIRECT`,
and reuses one 8 MiB aligned host staging buffer. Each chunk is converted as
needed and copied into its final tensor allocation. CUDA synchronization bounds
temporary lifetimes. Packed expert projections are filled in place; individual
expert tensors are views into these buffers, not copies. There is no buffered
I/O fallback for weights: a filesystem without direct-I/O support fails clearly.

Loading checks available system memory before reserving the weight set and keeps
16 GiB of system headroom. Validation commands additionally run under
`scripts/memory_guard.py`, which observes total system memory (including shared
GB10 GPU memory), aborts on new swap-out activity, and records peak consumption.
The guard's default growth ceiling is for BF16 validation, not a full FP32 model.

The actual LLaDA2.2-mini checkpoint has 32,511,286,784 bytes of BF16 weights
(30.28 GiB). Promoting only router matrices to FP32 adds about 19 MiB. At 131,072
tokens, the cache is:

```
20 layers × 2 (K,V) × 4 KV heads × 128 head dimension × 131072 tokens × 2 bytes
= 5 GiB
```

Attention workspaces and CUDA runtime allocations are additional. The server
defaults to the full 131,072-token request limit and admits one active generation.
Cache capacity follows each request's block-rounded prompt plus output budget.
Full-prefix diagnostic forwards are not the serving execution path.
Each attention tile is limited to 128 million score elements before allocation;
its query count shrinks at long contexts independently of the transformer batch.
Prefill defaults to 4,096-token batches and 1,024-query attention tiles. Transformer
batches cannot exceed 8,192 tokens, bounding routed-expert activations separately.
Attention reads the cache through views, without copying the committed prefix.
The full 128K serving limit has not been benchmarked.

## Reference validation

`scripts/layerwise_reference.py` imports the checkpoint's actual Python classes
but instantiates one transformer layer at a time on the meta device, streams its
weights directly into final tensors, evaluates it, and releases it before the
next layer. Even FP32 reference validation therefore requires only one layer's
weights (about 3 GiB for an MoE layer), rather than two 60.6 GiB model copies.
Reference diagnostics are limited to 512 input tokens to bound attention/logits.
The former complete-model `.from_pretrained(...).to('cuda')` path is removed.

For complete BF16 generation comparisons, `scripts/reference_generate.py` creates
the model on the meta device and assigns each directly loaded tensor into its
final parameter slot. There is one resident BF16 model, no complete intermediate
state dict, and no model-wide `.to(...)`. It shares the same exclusive lock.
Generation diagnostics are limited to 512 total tokens.

The first memory incident was caused by that former FP32 reference-loading path,
which could retain complete CPU and CUDA copies on the GB10's shared RAM. It was
not caused by a 128K KV cache; that validation used only 32 input tokens.

## Precision versus storage

BF16 weights do not require every arithmetic intermediate to be BF16. Efficient
kernels commonly use FP32 registers for nonlinearities, normalization reductions,
and attention softmax, while reading/writing BF16 tensors. For example, vLLM's
[SiLU implementation](https://github.com/vllm-project/vllm/blob/v0.10.2/csrc/activation_kernels.cu)
uses FP32 arithmetic within its activation kernel. Fusing casts/activation/multiply
removes temporary tensor traffic without requiring less accurate arithmetic.
minnow fuses SiLU/multiply: the activation uses FP32 registers, rounds to BF16,
then multiplies by the BF16 up-projection. An exhaustive finite-BF16 test verifies
bitwise equivalence with the explicit FP32 path. Expert gathering, FP32 weighting,
and reduction are fused with the same binary reduction tree and BF16 result.
Attention scaling, block masking, and FP32 softmax are fused for key lengths up to
8,192; longer keys use the bounded eager fallback. Scaled scores still round to
BF16 before softmax. The softmax reduction tree differs, allowing a BF16 rounding
step of numerical difference. RMSNorm retains explicit FP32 intermediates.
