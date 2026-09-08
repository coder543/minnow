// Native SM12x NVFP4 kernels. Fragment mapping follows the PTX ISA sections
// "Matrix Fragments for mma.m16n8k64" and "Block Scaling for mma.sync".
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>

// Dynamic per-token FP32 outer scale; 16-element blocks use E4M3 scales.
// Quantize once for all output columns, with no expanded weight intermediates.
extern "C" __global__ void minnow_nvfp4_quantize(const __nv_bfloat16* x,
    unsigned char* packed, int input, int rows) {
  unsigned char* codes=packed;
  unsigned char* scales=codes+size_t(rows)*input/2;
  float* globals=reinterpret_cast<float*>(scales+size_t(rows)*input/16);
  __shared__ float maxima[4];
  const int row=blockIdx.x, tid=threadIdx.x;
  float maximum=0.f;
  for(int col=tid;col<input;col+=128) maximum=fmaxf(maximum,fabsf(__bfloat162float(x[size_t(row)*input+col])));
  for(int d=16;d>0;d/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
  if(tid%32==0) maxima[tid/32]=maximum;
  __syncthreads();
  maximum=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
  const float global=maximum==0.f ? 1.f : fmaxf(maximum/2688.f,1.17549435e-38f);
  if(tid==0) globals[row]=global;
  for(int pair=tid;pair<input/2;pair+=128) {
    const int col=pair*2;
    float a=__bfloat162float(x[size_t(row)*input+col]);
    float b=__bfloat162float(x[size_t(row)*input+col+1]);
    float block=fmaxf(fabsf(a),fabsf(b));
    for(int d=4;d>0;d/=2) block=fmaxf(block,__shfl_xor_sync(0xffffffff,block,d,8));
    unsigned char scale=block==0.f ? 56 : __nv_cvt_float_to_fp8((block/6.f)/global,__NV_SATFINITE,__NV_E4M3);
    if(scale==0) scale=1;
    __nv_fp8_e4m3 sf; sf.__x=scale;
    const float divisor=float(sf)*global;
    codes[size_t(row)*input/2+pair]=__nv_cvt_float2_to_fp4x2(make_float2(a/divisor,b/divisor),__NV_E2M1,cudaRoundNearest);
    if(pair%8==0) scales[size_t(row)*input/16+pair/8]=scale;
  }
}
// Keep each row's BF16 pairs in registers across the amax reduction. Specialized
// model dimensions avoid a second global input pass and dynamically indexed arrays.
template<int Input>
__device__ void nvfp4_quantize_cached(const __nv_bfloat16* x,unsigned char* packed,int rows) {
  unsigned char* codes=packed;
  unsigned char* scales=codes+size_t(rows)*Input/2;
  float* globals=reinterpret_cast<float*>(scales+size_t(rows)*Input/16);
  __shared__ float maxima[4];
  const int row=blockIdx.x,tid=threadIdx.x;
  float a[Input/256],b[Input/256],maximum=0.f;
  #pragma unroll
  for(int i=0;i<Input/256;i++) {
    const __nv_bfloat162 pair=reinterpret_cast<const __nv_bfloat162*>(x+size_t(row)*Input)[tid+i*128];
    const float2 values=__bfloat1622float2(pair);
    a[i]=values.x;b[i]=values.y;
    maximum=fmaxf(maximum,fmaxf(fabsf(a[i]),fabsf(b[i])));
  }
  for(int d=16;d>0;d/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
  if(tid%32==0) maxima[tid/32]=maximum;
  __syncthreads();
  maximum=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
  const float global=maximum==0.f?1.f:fmaxf(maximum/2688.f,1.17549435e-38f);
  if(tid==0) globals[row]=global;
  #pragma unroll
  for(int i=0;i<Input/256;i++) {
    const int pair=tid+i*128;
    float block=fmaxf(fabsf(a[i]),fabsf(b[i]));
    for(int d=4;d>0;d/=2) block=fmaxf(block,__shfl_xor_sync(0xffffffff,block,d,8));
    unsigned char scale=block==0.f?56:__nv_cvt_float_to_fp8((block/6.f)/global,__NV_SATFINITE,__NV_E4M3);
    if(scale==0) scale=1;
    __nv_fp8_e4m3 sf;sf.__x=scale;
    const float divisor=float(sf)*global;
    codes[size_t(row)*Input/2+pair]=__nv_cvt_float2_to_fp4x2(make_float2(a[i]/divisor,b[i]/divisor),__NV_E2M1,cudaRoundNearest);
    if(pair%8==0) scales[size_t(row)*Input/16+pair/8]=scale;
  }
}
#define NVFP4_QUANTIZE(K) \
extern "C" __global__ void minnow_nvfp4_quantize_##K(const __nv_bfloat16* x,unsigned char* packed,int input,int rows) { \
  nvfp4_quantize_cached<K>(x,packed,rows); \
}
NVFP4_QUANTIZE(512)
NVFP4_QUANTIZE(1024)
NVFP4_QUANTIZE(2048)
NVFP4_QUANTIZE(4096)

__device__ __forceinline__ void nvfp4_mma(float* d,const unsigned* a,const unsigned* b,unsigned sa,unsigned sb) {
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
}
template<int Rows>
__device__ void nvfp4_gemm(const unsigned* x,const unsigned* xs,const float* xg,
    const unsigned* w,const unsigned* ws,const float* wg,const int* tiles,
    __nv_bfloat16* output,int input,int out,int indexed) {
  const int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
  const int e=tiles[blockIdx.y*3],row=tiles[blockIdx.y*3+1],rows=tiles[blockIdx.y*3+2];
  const int col=blockIdx.x*64+warp*16;
  const int* indices=tiles+gridDim.y*3;
  int sources[Rows/16][2];
  #pragma unroll
  for(int m=0;m<Rows/16;m++) {
    #pragma unroll
    for(int i=0;i<2;i++) {
      const int r=m*16+g+i*8;
      sources[m][i]=r<rows && indexed ? indices[row+r] : row+r;
    }
  }
  float acc[Rows/16][2][4]={};
  for(int k=0;k<input/64;k++) {
    unsigned bf[2][2],bs[2];
    #pragma unroll
    for(int n=0;n<2;n++) {
      const size_t tile=(size_t(e)*(out/8)+(col+n*8)/8)*(input/64)+k;
      if(col+n*8+g<out) {
        bf[n][0]=w[tile*64+lane]; bf[n][1]=w[tile*64+32+lane];
        bs[n]=ws[tile*8+g];
      } else { bf[n][0]=bf[n][1]=bs[n]=0; }
    }
    #pragma unroll
    for(int m=0;m<Rows/16;m++) {
      if(m*16>=rows) continue;
      unsigned af[4];
      #pragma unroll
      for(int i=0;i<4;i++) {
        const int r=m*16+g+(i%2)*8;
        const int source=sources[m][i%2];
        af[i]=r<rows ? x[size_t(source)*(input/8)+k*8+t+(i/2)*4] : 0;
      }
      const int r=m*16+g+(t%2)*8;
      const int source=sources[m][t%2];
      const unsigned scale=r<rows ? xs[size_t(source)*(input/64)+k] : 0;
      #pragma unroll
      for(int n=0;n<2;n++) nvfp4_mma(acc[m][n],af,bf[n],scale,bs[n]);
    }
  }
  #pragma unroll
  for(int m=0;m<Rows/16;m++) {
    #pragma unroll
    for(int i=0;i<4;i++) {
      const int r=m*16+g+(i/2)*8;
      if(r<rows) {
        const int source=sources[m][i/2];
        const float global=xg[source]*wg[e];
        #pragma unroll
        for(int n=0;n<2;n++) {
          const int c=col+n*8+t*2+i%2;
          if(c<out) output[size_t(row+r)*out+c]=__float2bfloat16_rn(acc[m][n][i]*global);
        }
      }
    }
  }
}
#define NVFP4_GEMM(R) \
extern "C" __global__ void minnow_nvfp4_gemm_##R(const unsigned char* packed, \
    const unsigned* w,const unsigned* ws,const float* wg,const int* tiles,__nv_bfloat16* output,int input,int out,int rows,int indexed) { \
  const unsigned* x=reinterpret_cast<const unsigned*>(packed); \
  const unsigned* xs=reinterpret_cast<const unsigned*>(packed+size_t(rows)*input/2); \
  const float* xg=reinterpret_cast<const float*>(packed+size_t(rows)*input*9/16); \
  nvfp4_gemm<R>(x,xs,xg,w,ws,wg,tiles,output,input,out,indexed); \
}
NVFP4_GEMM(16)
NVFP4_GEMM(32)
NVFP4_GEMM(64)
NVFP4_GEMM(128)
