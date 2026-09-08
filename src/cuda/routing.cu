// Two 16-row tiles per possible active expert, then 256 source indices and a
// in-place mixing plan. Integer descriptors occupy raw bits of FP32 storage.
__device__ void compact_route(const float* plan, float* output) {
  const int tid=threadIdx.x;
  int* descriptors=reinterpret_cast<int*>(output);
  const int id=(int)plan[49+tid];
  int row=0;
  for(int j=0;j<256;j++) {
    const int other=(int)plan[49+j];
    row += other<id || (other==id && j<tid);
  }
  descriptors[288+row]=tid/8;
  output[544+305+tid]=(float)row;
  if(tid<96) {
    const int rank=tid/2, part=tid%2;
    const int expert=rank<(int)plan[0] ? (int)plan[1+rank] : 256;
    int start=0,count=0;
    for(int j=0;j<256;j++) {
      const int other=(int)plan[49+j];
      start+=other<expert;
      count+=other==expert;
    }
    descriptors[tid*3]=expert==256?0:expert;
    descriptors[tid*3+1]=start+part*16;
    descriptors[tid*3+2]=max(0,min(16,count-part*16));
  }
}

// The mini decode route is one 32-token block: 256 experts, block capacity 48,
// top eight per token. All choices and mixing weights remain on the GPU.
// Plan: count, 48 sorted active expert IDs, 256 token expert IDs,
//       256 gathered row indices, 256 FP32 mixing weights.
__device__ void route_mini(
    const float* logits, const float* bias, float* plan, float scale) {
  __shared__ float scores[32 * 256];
  __shared__ float maxima[256];
  __shared__ int order[256];
  __shared__ int active[256];
  __shared__ int ranks[256];
  __shared__ int counts[8];
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  float maximum = -INFINITY;
  for (int t = 0; t < 32; ++t) {
    const float score = 1.0f / (1.0f + expf(-logits[t * 256 + tid]));
    scores[t * 256 + tid] = score;
    maximum = fmaxf(maximum, score + bias[tid]);
  }
  maxima[tid] = maximum;
  order[tid] = tid;
  active[tid] = 0;
  // Bitonic sort, descending score and then ascending expert ID.
  for (int size = 2; size <= 256; size <<= 1) {
    for (int step = size / 2; step > 0; step >>= 1) {
      __syncthreads();
      const int other = tid ^ step;
      const float value = maxima[other];
      const int id = order[other];
      const bool better = value > maxima[tid] || (value == maxima[tid] && id < order[tid]);
      const bool want_better = ((tid & size) == 0) == ((tid & step) == 0);
      const float own_value = maxima[tid];
      const int own_id = order[tid];
      __syncthreads();
      maxima[tid] = better == want_better ? value : own_value;
      order[tid] = better == want_better ? id : own_id;
    }
  }
  __syncthreads();
  for (int token = warp; token < 32; token += 8) {
    const int e0 = order[lane];
    const int e1 = lane < 16 ? order[lane + 32] : 0x7fffffff;
    float v0 = scores[token * 256 + e0] + bias[e0];
    float v1 = lane < 16 ? scores[token * 256 + e1] + bias[e1] : -INFINITY;
    float selected_scores[8];
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
      float value = fmaxf(v0, v1);
      int id = v0 > v1 ? e0 : (v1 > v0 ? e1 : min(e0, e1));
      #pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        const float other_value = __shfl_xor_sync(0xffffffff, value, mask);
        const int other_id = __shfl_xor_sync(0xffffffff, id, mask);
        if (other_value > value || (other_value == value && other_id < id)) {
          value = other_value;
          id = other_id;
        }
      }
      selected_scores[slot] = scores[token * 256 + id];
      if (lane == 0) {
        plan[49 + token * 8 + slot] = (float)id;
        atomicOr(&active[id], 1);
      }
      if (e0 == id) v0 = -INFINITY;
      if (e1 == id) v1 = -INFINITY;
    }
    if (lane == 0) {
      // Preserve the reference CPU route's sequential FP32 sum and division.
      float sum = 0.0f;
      #pragma unroll
      for (int slot = 0; slot < 8; ++slot) sum += selected_scores[slot];
      sum += 1e-20f;
      #pragma unroll
      for (int slot = 0; slot < 8; ++slot)
        plan[561 + token * 8 + slot] = selected_scores[slot] / sum * scale;
    }
  }
  __syncthreads();
  const unsigned mask = __ballot_sync(0xffffffff, active[tid] != 0);
  if (lane == 0) counts[warp] = __popc(mask);
  __syncthreads();
  int rank = __popc(mask & ((1u << lane) - 1));
  for (int w = 0; w < warp; ++w) rank += counts[w];
  if (active[tid]) {
    ranks[tid] = rank;
    plan[1 + rank] = (float)tid;
  }
  if (tid == 0) {
    int count = 0;
    for (int w = 0; w < 8; ++w) count += counts[w];
    plan[0] = (float)count;
  }
  __syncthreads();
  plan[305 + tid] = (float)(ranks[(int)plan[49 + tid]] * 32 + tid / 8);
  // Inactive slots are initialized too, for the cuBLAS comparison path.
  if (tid >= (int)plan[0] && tid < 48) plan[1 + tid] = plan[1];
}

extern "C" __global__ void minnow_expert_pointers(const float* plan,
    unsigned long long* pointers, unsigned long long x, unsigned long long w,
    unsigned long long y, int out_dim, int in_dim, int batched) {
  const int b = threadIdx.x;
  if (b < 48) {
    pointers[b] = w + (size_t)(int)plan[1 + b] * out_dim * in_dim * 2;
    pointers[48 + b] = x + (batched ? (size_t)b * 32 * in_dim * 2 : 0);
    pointers[96 + b] = y + (size_t)b * 32 * out_dim * 2;
  }
}

extern "C" __global__ void minnow_mix_routed(const __nv_bfloat16* experts,
    const float* plan, __nv_bfloat16* output, int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 32 * hidden) return;
  const int token = i / hidden, col = i % hidden;
  float v[8];
  #pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    const int assignment = token * 8 + slot;
    v[slot] = __fadd_rn(0.f, __bfloat162float(experts[(size_t)(int)plan[305 + assignment] * hidden + col]) * plan[561 + assignment]);
  }
  #pragma unroll
  for (int step = 4; step > 0; step >>= 1) {
    #pragma unroll
    for (int slot = 0; slot < step; ++slot) v[slot] += v[slot + step];
  }
  output[i] = __float2bfloat16_rn(v[0]);
}


extern "C" __global__ void minnow_route_mini(const float* logits,const float* bias,float* plan,float scale) {
  route_mini(logits,bias,plan,scale);
}
extern "C" __global__ void minnow_route_mini_compact(const float* logits,const float* bias,float* output,float scale) {
  float* plan=output+544;
  route_mini(logits,bias,plan,scale);
  __syncthreads();
  compact_route(plan,output);
}
