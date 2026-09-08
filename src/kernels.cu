#include <cuda_bf16.h>
#include <math.h>
#include <stddef.h>
#include "cuda/routing.cu"
#include "cuda/qkv.cu"
#include "cuda/quantized.cu"

// Preserve the reference's BF16 rounding BETWEEN SiLU and multiplication.
// FP32 values stay in registers; the only tensor output is BF16.
extern "C" __global__ void minnow_silu_mul_bf16(
    size_t n, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* output) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float x = __bfloat162float(gate[i]);
    const __nv_bfloat16 activated = __float2bfloat16_rn(x / (1.0f + expf(-x)));
    output[i] = __hmul(activated, up[i]);
  }
}

template <bool Max>
__device__ float block_reduce(float value, float* scratch) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  #pragma unroll
  for (int mask = 16; mask; mask >>= 1) {
    const float other = __shfl_xor_sync(0xffffffff, value, mask);
    value = Max ? fmaxf(value, other) : value + other;
  }
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < blockDim.x / 32 ? scratch[lane] : (Max ? -INFINITY : 0.0f);
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) {
      const float other = __shfl_xor_sync(0xffffffff, value, mask);
      value = Max ? fmaxf(value, other) : value + other;
    }
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

// Each block owns one attention row. Scaled scores and exponentials stay in
// FP32 registers; scaling still rounds to BF16 before softmax, like eager HF.
template <int Items>
__device__ void block_softmax(
    const __nv_bfloat16* scores, __nv_bfloat16* out, int cols, int queries,
    int offset, int block_size, float scale) {
  __shared__ float scratch[4];
  const size_t row = blockIdx.x;
  const int visible = min(cols, ((offset + (int)(row % queries)) / block_size + 1) * block_size);
  float values[Items];
  float maximum = -INFINITY;
  #pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int col = threadIdx.x + i * 128;
    const float value = col < visible
        ? __bfloat162float(__float2bfloat16_rn(__bfloat162float(scores[row * cols + col]) * scale))
        : -INFINITY;
    values[i] = value;
    maximum = fmaxf(maximum, value);
  }
  maximum = block_reduce<true>(maximum, scratch);
  // All threads consume the maximum before the next reduction reuses scratch.
  __syncthreads();
  float sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < Items; ++i) {
    values[i] = expf(values[i] - maximum);
    sum += values[i];
  }
  const float inv = 1.0f / block_reduce<false>(sum, scratch);
  #pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int col = threadIdx.x + i * 128;
    if (col < cols) out[row * cols + col] = __float2bfloat16_rn(values[i] * inv);
  }
}

#define SOFTMAX(Items) \
extern "C" __global__ void minnow_block_softmax_##Items( \
    const __nv_bfloat16* scores, __nv_bfloat16* out, int cols, int queries, \
    int offset, int block_size, float scale) { \
  block_softmax<Items>(scores, out, cols, queries, offset, block_size, scale); \
}
SOFTMAX(1)
SOFTMAX(2)
SOFTMAX(4)
SOFTMAX(8)
SOFTMAX(16)
SOFTMAX(32)
SOFTMAX(64)

// Long rows use bounded register storage. Re-read BF16 scores instead of
// allocating FP32 scores, a host-built mask, and intermediate softmax tensors.
extern "C" __global__ void minnow_block_softmax_long(
    const __nv_bfloat16* scores, __nv_bfloat16* out, int cols, int queries,
    int offset, int block_size, float scale) {
  __shared__ float scratch[8];
  const size_t row = blockIdx.x;
  const int visible = min(cols, ((offset + (int)(row % queries)) / block_size + 1) * block_size);
  float maximum = -INFINITY;
  for (int col = threadIdx.x; col < visible; col += blockDim.x) {
    const float value = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(scores[row * cols + col]) * scale));
    maximum = fmaxf(maximum, value);
  }
  maximum = block_reduce<true>(maximum, scratch);
  __syncthreads();
  float sum = 0.0f;
  for (int col = threadIdx.x; col < visible; col += blockDim.x) {
    const float value = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(scores[row * cols + col]) * scale));
    sum += expf(value - maximum);
  }
  const float inv = 1.0f / block_reduce<false>(sum, scratch);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float probability = 0.0f;
    if (col < visible) {
      const float value = __bfloat162float(__float2bfloat16_rn(
          __bfloat162float(scores[row * cols + col]) * scale));
      probability = expf(value - maximum) * inv;
    }
    out[row * cols + col] = __float2bfloat16_rn(probability);
  }
}

// Gather assigned expert rows, weight in FP32, and reduce in the same binary
// tree as Candle's sum. No [tokens, top_k, hidden] intermediate is materialized.
template <int K>
__device__ void mix_experts(size_t n, int hidden, int top_k,
    const __nv_bfloat16* experts, const unsigned int* rows,
    const float* weights, __nv_bfloat16* out) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const size_t first = i / hidden * top_k;
  const int col = i % hidden;
  float values[K];
  #pragma unroll
  for (int k = 0; k < K; ++k) {
    const float product = k < top_k
        ? __bfloat162float(experts[(size_t)rows[first + k] * hidden + col]) * weights[first + k]
        : 0.0f;
    // Candle initializes each summand accumulator to +0, including signed zero.
    values[k] = __fadd_rn(0.0f, product);
  }
  #pragma unroll
  for (int stride = K / 2; stride > 0; stride /= 2) {
    #pragma unroll
    for (int k = 0; k < stride; ++k) values[k] += values[k + stride];
  }
  out[i] = __float2bfloat16_rn(values[0]);
}
#define MIX(K) \
extern "C" __global__ void minnow_mix_experts_##K(size_t n, int hidden, int top_k, \
    const __nv_bfloat16* experts, const unsigned int* rows, const float* weights, __nv_bfloat16* out) { \
  mix_experts<K>(n, hidden, top_k, experts, rows, weights, out); \
}
MIX(1)
MIX(2)
MIX(4)
MIX(8)
MIX(16)
MIX(32)

// Split the very wide vocabulary across CTAs. A single warp per vocabulary row
// severely underutilizes the GPU for a 32-token diffusion block.
extern "C" __global__ void minnow_greedy_partials(
    const float* logits, float* partials, int cols, int tiles) {
  __shared__ float scratch[8];
  __shared__ int indices[256];
  const int start = blockIdx.x * 4096;
  const size_t row = blockIdx.y;
  float values[16];
  float maximum = -INFINITY;
  int index = 0x7fffffff;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int col = start + threadIdx.x + i * 256;
    const float v = col < cols ? logits[row * cols + col] : -INFINITY;
    values[i] = v;
    if (v > maximum || (v == maximum && col < index)) { maximum = v; index = col; }
  }
  const float global_max = block_reduce<true>(maximum, scratch);
  indices[threadIdx.x] = maximum == global_max ? index : 0x7fffffff;
  __syncthreads();
  for (int step = 128; step > 0; step >>= 1) {
    if (threadIdx.x < step) indices[threadIdx.x] = min(indices[threadIdx.x], indices[threadIdx.x + step]);
    __syncthreads();
  }
  float sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    // An all -infinity tile contributes zero, without inf-inf producing NaN.
    sum += global_max == -INFINITY ? 0.0f : expf(values[i] - global_max);
  }
  sum = block_reduce<false>(sum, scratch);
  if (threadIdx.x == 0) {
    const size_t out = (row * tiles + blockIdx.x) * 3;
    partials[out] = global_max;
    partials[out + 1] = sum;
    partials[out + 2] = (float)indices[0];
  }
}

extern "C" __global__ void minnow_greedy_finish(
    const float* partials, float* output, int tiles) {
  __shared__ float scratch[4];
  __shared__ int indices[128];
  const size_t base = (size_t)blockIdx.x * tiles * 3;
  float maximum = -INFINITY;
  int index = 0x7fffffff;
  for (int i = threadIdx.x; i < tiles; i += 128) {
    const float v = partials[base + i * 3];
    const int id = (int)partials[base + i * 3 + 2];
    if (v > maximum || (v == maximum && id < index)) { maximum = v; index = id; }
  }
  const float global_max = block_reduce<true>(maximum, scratch);
  indices[threadIdx.x] = maximum == global_max ? index : 0x7fffffff;
  __syncthreads();
  for (int step = 64; step > 0; step >>= 1) {
    if (threadIdx.x < step) indices[threadIdx.x] = min(indices[threadIdx.x], indices[threadIdx.x + step]);
    __syncthreads();
  }
  float sum = 0.0f;
  for (int i = threadIdx.x; i < tiles; i += 128) {
    const float m = partials[base + i * 3];
    sum += m == -INFINITY ? 0.0f : partials[base + i * 3 + 1] * expf(m - global_max);
  }
  sum = block_reduce<false>(sum, scratch);
  if (threadIdx.x == 0) {
    output[blockIdx.x * 2] = (float)indices[0];
    output[blockIdx.x * 2 + 1] = 1.0f / sum;
  }
}

// Reproduce Candle's FP32 reduction tree, sqrt/reciprocal, and BF16 rounding
// before the learned scale. Supports strided [outer, rows, hidden] inputs.
extern "C" __global__ void minnow_rms_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int hidden, int rows, size_t stride_outer,
    size_t stride_row, float eps) {
  __shared__ float sums[1024];
  const int tid = threadIdx.x;
  const size_t row = blockIdx.x;
  const size_t base = row / rows * stride_outer + row % rows * stride_row;
  float sum = 0.0f;
  for (int col = tid; col < hidden; col += blockDim.x) {
    const float x = __bfloat162float(input[base + col]);
    sum += x * x;
  }
  sums[tid] = sum;
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (tid < stride) sums[tid] += sums[tid + stride];
  }
  if (tid == 0) sums[0] = 1.0f / sqrtf(sums[0] * (1.0f / hidden) + eps);
  __syncthreads();
  const float inv = sums[0];
  for (int col = tid; col < hidden; col += blockDim.x) {
    const __nv_bfloat16 normalized = __float2bfloat16_rn(__bfloat162float(input[base + col]) * inv);
    output[row * hidden + col] = __hmul(normalized, weight[col]);
  }
}
