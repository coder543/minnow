// W4A8/W8A8 on SM80+. Integer accumulators are exact within each K group;
// FP32 activation and FP16 weight scales are applied before the next group.
// Packed checkpoint weights are expanded/reordered only in registers.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

template<int Group>
__device__ void quantize_a8(const __nv_bfloat16* x, unsigned char* codes, float* scales, int groups) {
  const int lane=threadIdx.x%32, item=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  if (item>=groups) return;
  constexpr int Items=(Group+31)/32;
  float values[Items], maximum=0.f;
  #pragma unroll
  for(int i=0;i<Items;i++) {
    const int k=lane+i*32;
    values[i]=k<Group?__bfloat162float(x[size_t(item)*Group+k]):0.f;
    maximum=fmaxf(maximum,fabsf(values[i]));
  }
  #pragma unroll
  for(int d=16;d>0;d/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
  const float scale=maximum==0.f?1.f:fmaxf(maximum/127.f,1.17549435e-38f);
  if(lane==0) scales[item]=scale;
  #pragma unroll
  for(int i=0;i<Items;i++) {
    const int k=lane+i*32;
    if(k<Group) codes[size_t(item)*Group+k]=static_cast<signed char>(
        __float2int_rn(fminf(127.f,fmaxf(-127.f,values[i]/scale))));
  }
}
#define QUANTIZE(G) \
extern "C" __global__ void minnow_quantize_a8_##G(const __nv_bfloat16* x,unsigned char* codes,float* scales,int groups) { \
  quantize_a8<G>(x,codes,scales,groups); \
}
QUANTIZE(16)
QUANTIZE(32)
QUANTIZE(64)
QUANTIZE(128)

__device__ __forceinline__ unsigned expand_i4(unsigned v) {
  v=(v & 15) | ((v & 0xf0)<<4) | ((v & 0xf00)<<8) | ((v & 0xf000)<<12);
  return v | ((v & 0x08080808u)*30u);
}
template<int Bits,bool Packed>
__device__ __forceinline__ unsigned weight_a8(const unsigned char* codes,
    int expert,int c,int input,int out,int k,int t) {
  if(c>=out) return 0;
  if constexpr(Packed) {
    const size_t tile=(size_t(expert)*(out/8)+c/8)*(input/16)+k/16;
    const size_t offset=(tile*32+(c%8)*4+(t%2)*2)*(Bits/2)+(t/2)*(Bits/4);
    if constexpr(Bits==8) {
      const unsigned a=*reinterpret_cast<const unsigned short*>(codes+offset);
      const unsigned b=*reinterpret_cast<const unsigned short*>(codes+offset+4);
      return a | (b<<16);
    } else return expand_i4(unsigned(codes[offset]) | (unsigned(codes[offset+2])<<8));
  } else {
    const size_t index=(size_t(expert)*out+c)*input+k+t*4;
    if constexpr(Bits==8) return *reinterpret_cast<const unsigned*>(codes+index);
    else return expand_i4(*reinterpret_cast<const unsigned short*>(codes+index/2));
  }
}
__device__ __forceinline__ void mma_i8(int* d,const unsigned* a,const unsigned* b,int group) {
  if(group==16) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(b[0]));
  } else {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  }
}
template<int Rows,int Bits,bool Packed,bool Pair,int Group=0>
__device__ void gemm_a8(const unsigned char* x,const float* x_scales,const unsigned char* codes,
    const __half* scales,const int* tiles,__nv_bfloat16* output,int input,int out,int group,int indexed) {
  if constexpr(Group>0) group=Group;
  const int warp=(threadIdx.x/32)%4,lane=threadIdx.x%32,g=lane/4,t=lane%4;
  const int expert=tiles[blockIdx.y*3],row=tiles[blockIdx.y*3+1],rows=tiles[blockIdx.y*3+2];
  if(rows==0) return;
  const int col=blockIdx.x*64+warp*16,groups=input/group;
  size_t source[Rows/16][2];
  #pragma unroll
  for(int m=0;m<Rows/16;m++) {
    #pragma unroll
    for(int i=0;i<2;i++) {
      const int r=m*16+g+i*8;
      source[m][i]=r<rows?(indexed?tiles[gridDim.y*3+row+r]:row+r):0;
    }
  }
  float acc[Rows/16][2][4]={};
  for(int block=0;block<groups;block++) {
    int sums[Rows/16][2][4]={};
    // Each group is at most 128 values; no INT32 overflow is possible.
    constexpr int Prefetch=Group==128?2:1;
    #pragma unroll
    for(int base=0;base<4;base+=Prefetch) {
      unsigned weights[Prefetch][2][2];
      #pragma unroll
      for(int h=0;h<Prefetch;h++) {
        if((base+h)*32>=group) continue;
        const int k=block*group+(base+h)*32;
        #pragma unroll
        for(int n=0;n<2;n++) {
          weights[h][n][0]=weight_a8<Bits,Packed>(codes,expert,col+n*8+g,input,out,k,t);
          weights[h][n][1]=group==16?0:weight_a8<Bits,Packed>(codes,expert,col+n*8+g,input,out,k+16,t);
        }
      }
      #pragma unroll
      for(int h=0;h<Prefetch;h++) {
      if((base+h)*32>=group) continue;
      const int k=block*group+(base+h)*32;
      #pragma unroll
      for(int m=0;m<Rows/16;m++) {
        if(m*16>=rows) continue;
        unsigned a[4];
        #pragma unroll
        for(int i=0;i<4;i++) {
          const int r=m*16+g+(i%2)*8,c=k+t*4+(i/2)*16;
          a[i]=(r<rows && (group!=16 || i<2))?
              *reinterpret_cast<const unsigned*>(x+source[m][i%2]*input+c):0;
        }
        #pragma unroll
        for(int n=0;n<2;n++) mma_i8(sums[m][n],a,weights[h][n],group);
      }
      }
    }
    #pragma unroll
    for(int n=0;n<2;n++) {
      float ws[2];
      #pragma unroll
      for(int j=0;j<2;j++) {
        const int c=col+n*8+t*2+j;
        const size_t si=Packed?(size_t(expert)*(out/8)+c/8)*groups*8+block*8+c%8:
            (size_t(expert)*out+c)*groups+block;
        ws[j]=c<out?__half2float(scales[si]):0.f;
      }
      #pragma unroll
      for(int m=0;m<Rows/16;m++) {
        #pragma unroll
        for(int i=0;i<4;i++) {
          const float scale=x_scales[source[m][i/2]*groups+block]*ws[i%2];
          acc[m][n][i]+=float(sums[m][n][i])*scale;
        }
      }
    }
  }
  #pragma unroll
  for(int m=0;m<Rows/16;m++) {
    #pragma unroll
    for(int n=0;n<2;n++) {
      #pragma unroll
      for(int i=0;i<4;i++) {
        const int r=m*16+g+(i/2)*8,c=col+n*8+t*2+i%2;
        if(r<rows && c<out) {
          const size_t dest=Pair?size_t(r)*64+warp*16+n*8+t*2+i%2:size_t(row+r)*out+c;
          output[dest]=__float2bfloat16_rn(acc[m][n][i]);
        }
      }
    }
  }
}
#define GEMM(Bits,Rows,Name,Packed,Group) \
extern "C" __global__ void minnow_a8_##Name##_##Bits##_##Rows(const unsigned char* x,const float* xs, \
    const unsigned char* codes,const __half* scales,const int* tiles,__nv_bfloat16* output,int input,int out,int group,int indexed) { \
  gemm_a8<Rows,Bits,Packed,false,Group>(x,xs,codes,scales,tiles,output,input,out,group,indexed); \
} \
extern "C" __global__ void minnow_a8_pair_##Name##_##Bits##_##Rows(const unsigned char* x,const float* xs, \
    const unsigned char* codes,const __half* scales,const int* tiles,__nv_bfloat16* output,int input,int out,int group,int indexed, \
    const unsigned char* up,const __half* us) { \
  __shared__ __nv_bfloat16 temp[2*Rows*64]; \
  const int part=threadIdx.x/128; \
  gemm_a8<Rows,Bits,Packed,true,Group>(x,xs,part?up:codes,part?us:scales,tiles,temp+part*Rows*64,input,out,group,indexed); \
  __syncthreads(); \
  const int row=tiles[blockIdx.y*3+1],rows=tiles[blockIdx.y*3+2]; \
  for(int i=threadIdx.x;i<Rows*64;i+=256) { \
    const int r=i/64,c=blockIdx.x*64+i%64; \
    if(r<rows && c<out) { \
      const float gate=__bfloat162float(temp[i]); \
      output[size_t(row+r)*out+c]=__hmul(__float2bfloat16_rn(gate/(1.f+expf(-gate))),temp[Rows*64+i]); \
    } \
  } \
}
#define VARIANTS(Bits,Rows) \
  GEMM(Bits,Rows,packed,true,0) GEMM(Bits,Rows,row,false,0) \
  GEMM(Bits,Rows,packed_g128,true,128) GEMM(Bits,Rows,row_g128,false,128)
VARIANTS(4,16)
VARIANTS(4,32)
VARIANTS(4,64)
VARIANTS(4,128)
VARIANTS(8,16)
VARIANTS(8,32)
VARIANTS(8,64)
VARIANTS(8,128)
