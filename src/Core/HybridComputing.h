#ifndef HYB_COMP_H
#define HYB_COMP_H

#ifdef __CUDACC__

#define __common __device__
#define __cmf_gpu 1
#define __cmf_cpu 0
#define HyCore HyCoreGPU

#else

#define __common
#define __cmf_gpu 0
#define __cmf_cpu 1
#define HyCore HyCoreCPU

#endif

#endif
