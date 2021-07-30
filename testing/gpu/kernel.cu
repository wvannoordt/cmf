#include <iostream>
#include "cmf.h"

__global__ void K_FillBlock(cmf::BlockArray<double, 1> arLb)
{
    
}

void FillBlockGpu(cmf::BlockArray<double, 1>& arLb)
{
    K_FillBlock<<<1,1>>>(arLb);
}

void tst(void)
{
    cmf::print("hello from CUDA kernel");
}