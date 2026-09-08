// Model-specific launch surface over FlashAttention's pipelined CUDA kernel.
#include "flash.h"
#include "flash_fwd_kernel.h"
using Traits=Flash_fwd_kernel_traits<128,64,64,4,false,false,cutlass::bfloat16_t>;
__device__ void minnow_flash_run(const __nv_bfloat16* q,const __nv_bfloat16* k,
    const __nv_bfloat16* v,__nv_bfloat16* output,int queries,int heads,int groups,int offset,
    size_t qh,size_t kh,size_t vh,float* lse) {
  Flash_fwd_params p={};
  p.q_ptr=const_cast<__nv_bfloat16*>(q);p.k_ptr=const_cast<__nv_bfloat16*>(k);p.v_ptr=const_cast<__nv_bfloat16*>(v);
  p.o_ptr=output;p.softmax_lse_ptr=lse;
  p.q_row_stride=p.k_row_stride=p.v_row_stride=128;
  p.q_head_stride=qh;p.k_head_stride=kh;p.v_head_stride=vh;
  p.o_row_stride=heads*128;p.o_head_stride=128;
  p.b=1;p.h=heads;p.h_k=heads/groups;p.h_h_k_ratio=groups;
  p.seqlen_q=queries;p.seqlen_k=queries+offset;p.d=p.d_rounded=128;p.total_q=queries;
  p.seqlen_q_rounded=(queries+127)/128*128;p.seqlen_k_rounded=(queries+offset+127)/128*128;
  p.scale_softmax=1.f;p.scale_softmax_log2=1.4426950408889634f;
  p.p_dropout=1.f;p.rp_dropout=1.f;p.window_size_left=-1;p.window_size_right=0;
  p.is_bf16=true;p.is_causal=true;p.is_seqlens_k_cumulative=true;
  flash::compute_attn_1rowblock<Traits,false,true,false,false,false,true,false,false>(p,0,blockIdx.z,blockIdx.x);
}
static_assert(Traits::kSmemSize==49152);

extern "C" __global__ void minnow_flash_block32(const __nv_bfloat16* q,
    const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* output,
    int queries,int heads,int groups,int offset,size_t qh,size_t kh,size_t vh,float* lse) {
  minnow_flash_run(q,k,v,output,queries,heads,groups,offset,qh,kh,vh,lse);
}
