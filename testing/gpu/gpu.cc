#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
#include "kernel.h"
#include "box.h"

using cmf::print;
using cmf::strformat;
using cmf::strunformat;
using cmf::cell_t;
using cmf::ZFill;

#define PI 3.141592655359

const double ghostJunkValue = -10.0;

void FillBlockCpu(cmf::BlockArray<double, 1>& arLb, const box& bx)
{
    for (cmf::cell_t k = arLb.kmin; k < arLb.kmax; k++)
    {
        for (cmf::cell_t j = arLb.jmin; j < arLb.jmax; j++)
        {
            for (cmf::cell_t i = arLb.imin; i < arLb.imax; i++)
            {
                arLb(0, i, j, k) = i;
                arLb(1, i, j, k) = j;
                arLb(2, i, j, k) = k;
                
                arLb(3, i, j, k) = bx.xmin + bx.dx*((double)i+0.5);
                arLb(4, i, j, k) = bx.ymin + bx.dy*((double)j+0.5);
                arLb(5, i, j, k) = bx.zmin + bx.dz*((double)k+0.5);
            }
        }
    }
}

void FillArr(cmf::CartesianMeshArray& arr)
{
    for (auto lb: arr)
    {
        auto info = arr.GetBlockInfo(lb);
        box bx;
        bx.xmin = info.blockBounds[0];
        bx.ymin = info.blockBounds[2];
        bx.zmin = info.blockBounds[2*(CMF_DIM-1)];
        bx.dx   = info.dx[0];
        bx.dy   = info.dx[1];
        bx.dz   = info.dx[CMF_DIM-1];
        auto device = arr.GetBlockDevice(lb);
        cmf::BlockArray<double, 1> arLb = arr[lb];
        if (device.isGpu) { FillBlockGpu(arLb, bx); }
        else { FillBlockCpu(arLb, bx); }
    }
    CMF_CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    std::string inFile = "input.ptl";
    cmf::ReadInput(inFile);
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CreateParallelContext(&argc, &argv);
    PTL::PropertyTree localTree;
    localTree.Read(inFile);
    cmf::CartesianMeshInputInfo inputInfo(localTree["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    
    auto& var = domain.DefineVariable("preData", cmf::CmfArrayType::CmfDouble, {6});
    
    var.ComponentName(0) = "i";
    var.ComponentName(1) = "j";
    var.ComponentName(2) = "k";
    var.ComponentName(3) = "x";
    var.ComponentName(4) = "y";
    var.ComponentName(5) = "z";
    
    FillArr(var);
    
    var.Exchange();
    var.ExportFile("output", "test");
    
    return 0;
}