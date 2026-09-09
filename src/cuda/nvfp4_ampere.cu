// NVFP4 arithmetic on SM80+ BF16 tensor cores. Activations are quantized once
// into exact E2M1*E4M3 BF16 operands; only this small activation buffer expands.
// Checkpoint weights retain the native FP4 layout and expand in registers.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

__device__ __forceinline__ float fp4_round(float x) {
  const float a=fabsf(x);
  const float value=a<=0.25f?0.f:a<0.75f?0.5f:a<=1.25f?1.f:a<1.75f?1.5f:
      a<=2.5f?2.f:a<3.5f?3.f:a<=5.f?4.f:6.f;
  return copysignf(value,x);
}
template<int Input=0>
__device__ void quantize(const __nv_bfloat16* x,unsigned char* output,int input,int rows) {
  if constexpr (Input>0) input=Input;
  auto* operands=reinterpret_cast<__nv_bfloat162*>(output);
  auto* globals=reinterpret_cast<float*>(output+size_t(rows)*input*2);
  __shared__ float maxima[4];
  const int row=blockIdx.x,tid=threadIdx.x;
  constexpr int Items=Input>0?Input/256:1;
  float2 values[Items];
  float maximum=0.f;
  if constexpr (Input>0) {
    #pragma unroll
    for(int i=0;i<Items;i++) {
      values[i]=__bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(x+size_t(row)*input)[tid+i*128]);
      maximum=fmaxf(maximum,fmaxf(fabsf(values[i].x),fabsf(values[i].y)));
    }
  } else {
    for(int col=tid;col<input;col+=128) maximum=fmaxf(maximum,fabsf(__bfloat162float(x[size_t(row)*input+col])));
  }
  for(int d=16;d>0;d/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
  if(tid%32==0) maxima[tid/32]=maximum;
  __syncthreads();
  maximum=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
  const float global=maximum==0.f?1.f:fmaxf(maximum/2688.f,1.17549435e-38f);
  if(tid==0) globals[row]=global;
  for(int pair=tid,i=0;pair<input/2;pair+=128,i++) {
    float2 v;
    if constexpr (Input>0) v=values[i];
    else v=__bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(x+size_t(row)*input)[pair]);
    float block=fmaxf(fabsf(v.x),fabsf(v.y));
    for(int d=4;d>0;d/=2) block=fmaxf(block,__shfl_xor_sync(0xffffffff,block,d,8));
    unsigned char scale=block==0.f?56:__nv_cvt_float_to_fp8((block/6.f)/global,__NV_SATFINITE,__NV_E4M3);
    if(scale==0) scale=1;
    __nv_fp8_e4m3 sf; sf.__x=scale;
    const float s=float(sf),divisor=s*global;
    operands[size_t(row)*input/2+pair]=__floats2bfloat162_rn(fp4_round(v.x/divisor)*s,fp4_round(v.y/divisor)*s);
  }
}
extern "C" __global__ void minnow_nvfp4_quantize(const __nv_bfloat16* x,unsigned char* packed,int input,int rows) {
  quantize(x,packed,input,rows);
}
#define QUANTIZE(K) \
extern "C" __global__ void minnow_nvfp4_quantize_##K(const __nv_bfloat16* x,unsigned char* packed,int input,int rows) { \
  quantize<K>(x,packed,input,rows); \
}
QUANTIZE(512)
QUANTIZE(1024)
QUANTIZE(2048)
QUANTIZE(4096)

__device__ __forceinline__ float fp4_value(unsigned code) {
  const unsigned magnitude=code&7;
  const float value=magnitude<2 ? 0.5f*float(magnitude)
      : __uint_as_float(((magnitude/2+126)<<23)|((magnitude&1)<<22));
  return code&8 ? -value : value;
}
__device__ __forceinline__ unsigned weight_pair(unsigned codes,float scale) {
  return unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(fp4_value(codes&15)*scale)))
      | (unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(fp4_value((codes>>4)&15)*scale)))<<16);
}
template<int Rows,bool Pair=false>
__device__ void gemm(const __nv_bfloat16* x,const float* xg,
    const unsigned* w,const unsigned char* ws,const float* wg,const int* tiles,
    __nv_bfloat16* output,int input,int out,int indexed) {
  const int lane=threadIdx.x%32,warp=(threadIdx.x/32)%4,g=lane/4,t=lane%4;
  const int e=tiles[blockIdx.y*3],row=tiles[blockIdx.y*3+1],rows=tiles[blockIdx.y*3+2];
  if(rows==0) return;
  const int col=blockIdx.x*64+warp*16;
  int sources[Rows/16][2];
  #pragma unroll
  for(int m=0;m<Rows/16;m++) {
    #pragma unroll
    for(int i=0;i<2;i++) {
      const int r=m*16+g+i*8;
      sources[m][i]=r<rows && indexed ? tiles[gridDim.y*3+row+r] : row+r;
    }
  }
  float acc[Rows/16][2][4]={};
  for(int k=0;k<input;k+=16) {
    unsigned bf[2][2];
    #pragma unroll
    for(int n=0;n<2;n++) {
      const size_t tile=(size_t(e)*(out/8)+(col+n*8)/8)*(input/64)+k/64;
      float scale=0.f;
      if(col+n*8+g<out) { __nv_fp8_e4m3 sf;sf.__x=ws[tile*32+g*4+k%64/16];scale=float(sf); }
      #pragma unroll
      for(int i=0;i<2;i++) {
        const unsigned codes=col+n*8+g<out?w[tile*64+k%64/32*32+g*4+k%32/8+i]:0;
        bf[n][i]=weight_pair(codes>>(t*8),scale);
      }
    }
    #pragma unroll
    for(int m=0;m<Rows/16;m++) {
      if(m*16>=rows) continue;
      unsigned af[4];
      #pragma unroll
      for(int i=0;i<4;i++) {
        const int r=m*16+g+(i%2)*8;
        af[i]=r<rows?*reinterpret_cast<const unsigned*>(x+size_t(sources[m][i%2])*input+k+t*2+(i/2)*8):0;
      }
      #pragma unroll
      for(int n=0;n<2;n++) {
        float* d=acc[m][n];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            : "r"(af[0]),"r"(af[1]),"r"(af[2]),"r"(af[3]),"r"(bf[n][0]),"r"(bf[n][1]));
      }
    }
  }
  #pragma unroll
  for(int m=0;m<Rows/16;m++) {
    #pragma unroll
    for(int i=0;i<4;i++) {
      const int r=m*16+g+(i/2)*8;
      if(r<rows) {
        const float global=xg[sources[m][i/2]]*wg[e];
        #pragma unroll
        for(int n=0;n<2;n++) {
          const int c=col+n*8+t*2+i%2;
          if(c<out) {
            const size_t dest=Pair?size_t(r)*64+c-blockIdx.x*64:size_t(row+r)*out+c;
            output[dest]=__float2bfloat16_rn(acc[m][n][i]*global);
          }
        }
      }
    }
  }
}
#define GEMM(R) \
extern "C" __global__ void minnow_nvfp4_gemm_##R(const unsigned char* packed, \
    const unsigned* w,const unsigned char* ws,const float* wg,const int* tiles,__nv_bfloat16* output,int input,int out,int rows,int indexed) { \
  gemm<R>(reinterpret_cast<const __nv_bfloat16*>(packed),reinterpret_cast<const float*>(packed+size_t(rows)*input*2),w,ws,wg,tiles,output,input,out,indexed); \
} \
extern "C" __global__ void minnow_nvfp4_pair_##R(const unsigned char* packed, \
    const unsigned* w,const unsigned char* ws,const float* wg,const int* tiles,__nv_bfloat16* output,int input,int out,int rows,int indexed, \
    const unsigned* up,const unsigned char* ups,const float* upg) { \
  __shared__ __nv_bfloat16 temp[2*R*64]; \
  const int count=tiles[blockIdx.y*3+2],start=tiles[blockIdx.y*3+1]; \
  if(count==0) return; \
  const int part=threadIdx.x/128; \
  gemm<R,true>(reinterpret_cast<const __nv_bfloat16*>(packed),reinterpret_cast<const float*>(packed+size_t(rows)*input*2), \
      part?up:w,part?ups:ws,part?upg:wg,tiles,temp+part*R*64,input,out,indexed); \
  __syncthreads(); \
  for(int i=threadIdx.x;i<count*64;i+=256) { \
    const int col=blockIdx.x*64+i%64; \
    if(col<out) { \
      const float g=__bfloat162float(temp[i]); \
      const __nv_bfloat16 activated=__float2bfloat16_rn(g/(1.f+expf(-g))); \
      output[size_t(start+i/64)*out+col]=__hmul(activated,temp[R*64+i]); \
    } \
  } \
}
GEMM(16)
GEMM(32)
GEMM(64)
GEMM(128)
