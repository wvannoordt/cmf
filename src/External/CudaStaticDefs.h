#ifndef CMF_CUDA_STATIC_DEFS_H
#define CMF_CUDA_STATIC_DEFS_H

#ifdef __CUDACC__
#   define _CmfShared_ __device__ __host__ 
#else
#   define _CmfShared_
#endif

#endif