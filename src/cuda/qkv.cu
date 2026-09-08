extern "C" __global__ void minnow_prepare_qkv(const __nv_bfloat16* input,
    const __nv_bfloat16* qweight, const __nv_bfloat16* kweight,
    const __nv_bfloat16* cos, const __nv_bfloat16* sin,
    __nv_bfloat16* output, int tokens, int query_heads, float eps) {
  __shared__ float sums[128];
  __shared__ __nv_bfloat16 normalized[128];
  const int col = threadIdx.x;
  const int head = blockIdx.x / tokens, token = blockIdx.x % tokens;
  const int base = (token * (query_heads + 8) + head) * 128;
  __nv_bfloat16 value = input[base + col];
  if (head < query_heads + 4) {
    const float v = __bfloat162float(value);
    sums[col] = __fadd_rn(0.f, v * v);
    for (int step = 64; step > 0; step >>= 1) {
      __syncthreads();
      if (col < step) sums[col] += sums[col + step];
    }
    if (col == 0) sums[0] = 1.f / sqrtf(sums[0] * (1.f / 128) + eps);
    __syncthreads();
    const __nv_bfloat16* weight = head < query_heads ? qweight : kweight;
    normalized[col] = __hmul(__float2bfloat16_rn(v * sums[0]), weight[col]);
    __syncthreads();
    value = normalized[col];
    if (col < 64) {
      const __nv_bfloat16 partner = col < 32 ? __hneg(normalized[col + 32]) : normalized[col - 32];
      // Explicit rounding prevents the driver's PTX JIT from contracting BF16
      // mul+add on newer targets. Each product must round before the sum.
      value = __hadd_rn(__hmul_rn(value, cos[token * 64 + col]), __hmul_rn(partner, sin[token * 64 + col]));
    }
  }
  output[(head * tokens + token) * 128 + col] = value;
}
