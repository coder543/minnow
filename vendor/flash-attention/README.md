# FlashAttention headers

Headers from `candle-flash-attn` 0.11.0 (Hugging Face Candle), derived from
Tri Dao's FlashAttention. The upstream copyright notices and BSD license are
retained; Candle additions are used under Apache-2.0. The CUTLASS license is
included for binary distributions. CUTLASS is fetched at the pinned commit in `build.rs` using CudaForge.

Minnow specializes the forward kernel for BF16, head dimension 128, and no
training/dropout. Local changes round the causal boundary up to a 32-token
block in `mask.h`, and retain BF16 score/scaled-score rounding before softmax
in `flash_fwd_kernel.h`. The launch wrapper lives in `src/cuda/flash.cu`.
