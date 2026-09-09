#include <mma.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float minnow_decode_weight(const unsigned char* codes, size_t index) {
  return float(reinterpret_cast<const signed char*>(codes)[index]);
}

__device__ void minnow_quant_gemm(const __nv_bfloat16* x, const unsigned char* codes,
    const __half* scales, const int* tiles, __nv_bfloat16* output, int input, int out, int group) {
  using namespace nvcuda;
  constexpr int M=32, N=64, K=32;
  __shared__ __align__(32) __nv_bfloat16 a[M*K];
  __shared__ __align__(32) __nv_bfloat16 b[N*K];
  __shared__ __align__(32) float c[M*N];
  const int tid=threadIdx.x, warp=tid/32;
  const int expert=tiles[blockIdx.y*3], row=tiles[blockIdx.y*3+1], rows=tiles[blockIdx.y*3+2];
  const int col=blockIdx.x*N;
  const int group_shift=__ffs(group)-1;
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[2];
  wmma::fill_fragment(acc[0],0.f); wmma::fill_fragment(acc[1],0.f);
  for (int start=0;start<input;start+=K) {
    for (int i=tid;i<M*K;i+=128) {
      const int r=i/K, k=start+i%K;
      a[i]=(r<rows && k<input)?x[size_t(row+r)*input+k]:__float2bfloat16_rn(0.f);
    }
    for (int i=tid;i<N*K;i+=128) {
      const int n=col+i/K,k=start+i%K;
      float value=0.f;
      if (n<out && k<input) {
        const size_t index=(size_t(expert)*out+n)*input+k;
        value=minnow_decode_weight(codes,index)*__half2float(scales[index>>group_shift]);
      }
      // Dequantization remains on chip, followed by BF16 tensor-core operands.
      b[i]=__float2bfloat16_rn(value);
    }
    __syncthreads();
    #pragma unroll
    for (int k=0;k<K;k+=16) {
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bf;
      wmma::load_matrix_sync(bf,b+warp*16*K+k,K);
      #pragma unroll
      for (int m=0;m<2;m++) {
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
        wmma::load_matrix_sync(af,a+m*16*K+k,K);
        wmma::mma_sync(acc[m],af,bf,acc[m]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int m=0;m<2;m++) wmma::store_matrix_sync(c+m*16*N+warp*16,acc[m],N,wmma::mem_row_major);
  __syncthreads();
  for (int i=tid;i<M*N;i+=128) {
    const int r=i/N,n=col+i%N;
    if (r<rows && n<out) output[size_t(row+r)*out+n]=__float2bfloat16_rn(c[i]);
  }
}
#define MINNOW_QUANT_GEMM(Bits) \
extern "C" __global__ void minnow_quant_gemm_##Bits(const __nv_bfloat16* x, const unsigned char* codes, \
    const __half* scales, const int* tiles, __nv_bfloat16* output, int input, int out, int group) { \
  minnow_quant_gemm(x,codes,scales,tiles,output,input,out,group); \
}
MINNOW_QUANT_GEMM(8)

__device__ __forceinline__ unsigned minnow_bf16_pair(float a, float b) {
  return unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(a)))
      | (unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(b)))<<16);
}
template<int Bits=8>
__device__ __forceinline__ unsigned minnow_weight_pair(const unsigned char* codes,
    float scale, size_t index) {
  if constexpr (Bits==4) {
    const unsigned pair=codes[index/2];
    return minnow_bf16_pair(float((int(pair & 15)^8)-8)*scale,
        float((int(pair>>4)^8)-8)*scale);
  }
  const unsigned pair=*reinterpret_cast<const unsigned short*>(codes+index);
  const float a=float(static_cast<signed char>(pair & 255));
  const float b=float(static_cast<signed char>(pair >> 8));
  return minnow_bf16_pair(a*scale,b*scale);
}
__device__ __forceinline__ void minnow_mma(float* d, const unsigned* a, const unsigned* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

// Register-only dequantization. A weight fragment is reused across every
// 16-row fragment in the tile. The default packed tile is 16 rows; larger
// variants remain available for hardware-specific benchmarking.
template<int Rows,bool Packed=false,bool Pair=false,int Prefetch=4,int Bits=8>
__device__ void minnow_quant_mma(const __nv_bfloat16* x,const unsigned char* codes,
    const __half* scales,const int* tiles,__nv_bfloat16* output,int input,int out,int group,int indexed=0) {
  const int warp=(threadIdx.x/32)%4,lane=threadIdx.x%32;
  const int g=lane/4,t=lane%4;
  const int expert=tiles[blockIdx.y*3],row=tiles[blockIdx.y*3+1],rows=tiles[blockIdx.y*3+2];
  if (rows==0) return;
  const int col=blockIdx.x*64+warp*16;
  size_t source[Rows/16][2];
  #pragma unroll
  for (int m=0;m<Rows/16;m++) {
    #pragma unroll
    for (int i=0;i<2;i++) {
      const int r=m*16+g+i*8;
      source[m][i]=size_t(r<rows ? (indexed ? tiles[gridDim.y*3+row+r] : row+r) : 0)*input;
    }
  }
  const int shift=__ffs(group)-1;
  float acc[Rows/16][2][4]={};
  // Issue independent packed loads ahead of the MMA chain. Four K fragments
  // overlap memory latency without changing the accumulation order.
  constexpr int Steps=Packed ? Prefetch : 1;
  for (int base=0;base<input;base+=16*Steps) {
    unsigned raw[Steps][2];
    float scale_values[Steps][2];
    if constexpr (Packed) {
      #pragma unroll
      for (int h=0;h<Steps;h++) {
        #pragma unroll
        for (int n=0;n<2;n++) {
          const int k=base+h*16;
          const size_t tile=(size_t(expert)*(out/8)+(col+n*8)/8);
          raw[h][n]=0; scale_values[h][n]=0.f;
          if (col+n*8+g<out && k<input) {
            const size_t offset=(tile*(input/16)+k/16)*32+lane;
            if constexpr (Bits==4) raw[h][n]=reinterpret_cast<const unsigned short*>(codes)[offset];
            else raw[h][n]=reinterpret_cast<const unsigned*>(codes)[offset];
            scale_values[h][n]=__half2float(scales[(tile*(input>>shift)+(k>>shift))*8+g]);
          }
        }
      }
    }
    #pragma unroll
    for (int h=0;h<Steps;h++) {
      const int k=base+h*16;
      if (k>=input) continue;
      unsigned bf[2][2];
      #pragma unroll
      for (int n=0;n<2;n++) {
        const int c=col+n*8+g;
        bf[n][0]=bf[n][1]=0;
        if (c<out) {
          if constexpr (Packed) {
            const float scale=scale_values[h][n];
            const unsigned values=raw[h][n];
            if constexpr (Bits==4) {
              bf[n][0]=minnow_bf16_pair(float((int(values & 15)^8)-8)*scale,
                  float((int((values>>4)&15)^8)-8)*scale);
              bf[n][1]=minnow_bf16_pair(float((int((values>>8)&15)^8)-8)*scale,
                  float((int(values>>12)^8)-8)*scale);
            } else {
            bf[n][0]=minnow_bf16_pair(float(static_cast<signed char>(values & 255))*scale,
                float(static_cast<signed char>((values>>8) & 255))*scale);
            bf[n][1]=minnow_bf16_pair(float(static_cast<signed char>((values>>16) & 255))*scale,
                float(static_cast<signed char>(values>>24))*scale);
            }
          } else {
            const size_t index=(size_t(expert)*out+c)*input+k+t*2;
            const float scale=__half2float(scales[index>>shift]);
            bf[n][0]=minnow_weight_pair<Bits>(codes,scale,index);
            bf[n][1]=minnow_weight_pair<Bits>(codes,scale,index+8);
          }
        }
      }
      #pragma unroll
      for (int m=0;m<Rows/16;m++) {
        if (m*16>=rows) continue;
        unsigned af[4];
        #pragma unroll
        for (int i=0;i<4;i++) {
          const int r=m*16+g+(i%2)*8;
          const int c=k+t*2+(i/2)*8;
          af[i]=r<rows ? *reinterpret_cast<const unsigned*>(x+source[m][i%2]+c):0u;
        }
        #pragma unroll
        for (int n=0;n<2;n++) minnow_mma(acc[m][n],af,bf[n]);
      }
    }
  }
  #pragma unroll
  for (int m=0;m<Rows/16;m++) {
    #pragma unroll
    for (int n=0;n<2;n++) {
      #pragma unroll
      for (int i=0;i<4;i++) {
        const int r=m*16+g+(i/2)*8,c=col+n*8+t*2+i%2;
        if (r<rows && c<out) {
          const size_t dest=Pair ? size_t(r)*64+warp*16+n*8+t*2+i%2 : size_t(row+r)*out+c;
          output[dest]=__float2bfloat16_rn(acc[m][n][i]);
        }
      }
    }
  }
}
#define MINNOW_QUANT_MMA(Bits,Rows) \
extern "C" __global__ void minnow_quant_mma_##Bits##_##Rows(const __nv_bfloat16* x,const unsigned char* codes, \
    const __half* scales,const int* tiles,__nv_bfloat16* output,int input,int out,int group) { \
  minnow_quant_mma<Rows,false,false,4,Bits>(x,codes,scales,tiles,output,input,out,group); \
}
MINNOW_QUANT_MMA(8,16)
MINNOW_QUANT_MMA(8,32)
MINNOW_QUANT_MMA(8,64)
MINNOW_QUANT_MMA(8,128)
MINNOW_QUANT_MMA(4,16)
MINNOW_QUANT_MMA(4,32)
MINNOW_QUANT_MMA(4,64)
MINNOW_QUANT_MMA(4,128)

#define MINNOW_QUANT_PACKED(Bits,Rows) \
extern "C" __global__ void minnow_quant_packed_##Bits##_##Rows(const __nv_bfloat16* x,const unsigned char* codes, \
    const __half* scales,const int* tiles,__nv_bfloat16* output,int input,int out,int group) { \
  minnow_quant_mma<Rows,true,false,4,Bits>(x,codes,scales,tiles,output,input,out,group); \
}
MINNOW_QUANT_PACKED(8,16)
MINNOW_QUANT_PACKED(8,32)
MINNOW_QUANT_PACKED(8,64)
MINNOW_QUANT_PACKED(8,128)
MINNOW_QUANT_PACKED(4,16)
MINNOW_QUANT_PACKED(4,32)
MINNOW_QUANT_PACKED(4,64)
MINNOW_QUANT_PACKED(4,128)

// The two projections retain their BF16 rounding before SiLU and multiplication.
// Their intermediate tiles stay in shared memory and never enter global storage.
#define MINNOW_QUANT_INDEXED(Rows,Name,Packed,Prefetch,Bits) \
extern "C" __global__ void minnow_quant_indexed_##Name##_##Rows(const __nv_bfloat16* x,const unsigned char* codes, \
    const __half* scales,const int* tiles,__nv_bfloat16* output,int input,int out,int group,int indexed) { \
  minnow_quant_mma<Rows,Packed,false,Prefetch,Bits>(x,codes,scales,tiles,output,input,out,group,indexed); \
} \
extern "C" __global__ void minnow_quant_pair_##Name##_##Rows(const __nv_bfloat16* x,const unsigned char* codes, \
    const __half* scales,const int* tiles,__nv_bfloat16* output,int input,int out,int group,int indexed, \
    const unsigned char* up,const __half* up_scales) { \
  __shared__ __nv_bfloat16 temp[2*Rows*64]; \
  const int part=threadIdx.x/128; \
  minnow_quant_mma<Rows,Packed,true,Prefetch,Bits>(x,part?up:codes,part?up_scales:scales,tiles,temp+part*Rows*64,input,out,group,indexed); \
  __syncthreads(); \
  const int row=tiles[blockIdx.y*3+1],rows=tiles[blockIdx.y*3+2]; \
  for (int i=threadIdx.x;i<Rows*64;i+=256) { \
    const int r=i/64,c=blockIdx.x*64+i%64; \
    if (r<rows && c<out) { \
      const float g=__bfloat162float(temp[i]); \
      const __nv_bfloat16 activated=__float2bfloat16_rn(g/(1.f+expf(-g))); \
      output[size_t(row+r)*out+c]=__hmul(activated,temp[Rows*64+i]); \
    } \
  } \
}
MINNOW_QUANT_INDEXED(16,packed,true,4,8)
MINNOW_QUANT_INDEXED(32,packed,true,4,8)
MINNOW_QUANT_INDEXED(64,packed,true,4,8)
MINNOW_QUANT_INDEXED(128,packed,true,4,8)
MINNOW_QUANT_INDEXED(16,mma,false,4,8)
MINNOW_QUANT_INDEXED(32,mma,false,4,8)
MINNOW_QUANT_INDEXED(64,mma,false,4,8)
MINNOW_QUANT_INDEXED(128,mma,false,4,8)
MINNOW_QUANT_INDEXED(16,packed4,true,2,4)
MINNOW_QUANT_INDEXED(32,packed4,true,4,4)
MINNOW_QUANT_INDEXED(64,packed4,true,4,4)
MINNOW_QUANT_INDEXED(128,packed4,true,4,4)
MINNOW_QUANT_INDEXED(16,mma4,false,4,4)
MINNOW_QUANT_INDEXED(32,mma4,false,4,4)
MINNOW_QUANT_INDEXED(64,mma4,false,4,4)
MINNOW_QUANT_INDEXED(128,mma4,false,4,4)

// SM86 benefits from a shorter load pipeline. Keep the original four-fragment
// variants intact for other architectures, including GB10/SM121.
MINNOW_QUANT_INDEXED(16,ampere,true,2,8)
extern "C" __global__ void minnow_quant_ampere_16(const __nv_bfloat16* x,const unsigned char* codes,
    const __half* scales,const int* tiles,__nv_bfloat16* output,int input,int out,int group) {
  minnow_quant_mma<16,true,false,2>(x,codes,scales,tiles,output,input,out,group);
}
