#include <iostream>
#include "cmf.h"

using cmf::print;

__global__ void K_FillBlock(cmf::BlockArray<double, 1> arLb)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;
    int k = blockIdx.z*blockDim.z+threadIdx.z;
    if (i>=arLb.imin && i<arLb.imax && j>=arLb.jmin && j<arLb.jmax && k>=arLb.kmin && k<arLb.kmax)
    {
        arLb(0, i, j, k) = i;
        arLb(1, i, j, k) = j;
        arLb(2, i, j, k) = k;
    }
}

void FillBlockGpu(cmf::BlockArray<double, 1>& arLb)
{
    int nx = arLb.imax-arLb.imin;
    int ny = arLb.jmax-arLb.jmin;
    int nz = arLb.kmax-arLb.kmin;
    dim3 block(4,4,4);
    dim3 grid (0,0,0);
    grid.x = (nx + block.x - 1)/block.x;
    grid.y = (ny + block.y - 1)/block.y;
    grid.z = (nz + block.z - 1)/block.z;
    K_FillBlock<<<grid, block>>>(arLb);
}