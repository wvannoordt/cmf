#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
#include "kernel.h"

using cmf::print;
using cmf::strformat;
using cmf::strunformat;
using cmf::cell_t;
using cmf::ZFill;

#define PI 3.141592655359

const double ghostJunkValue = -10.0;

void FillBlock(cmf::BlockArray<double, 1>& arLb)
{
    for (cell_t k = arLb.kmin; k < arLb.kmax; k++)
    {
        for (cell_t j = arLb.jmin; j < arLb.jmax; j++)
        {
            for (cell_t i = arLb.imin; i < arLb.imax; i++)
            {
                for (int v = 0; v < arLb.dims[0]; v++)
                {
                    arLb(v, i, j, k) = (double)((3-2*v)*(i+j+k));
                }
            }
        }
    }
}

void FillArr(cmf::CartesianMeshArray& arr)
{
    for (auto lb: arr)
    {
        cmf::BlockArray<double, 1> arLb = arr[lb];
        FillBlock(arLb);
    }
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
    auto& var = domain.DefineVariable("preData", cmf::CmfArrayType::CmfDouble, {3});
    
    var.ComponentName({0}) = "data";
    var.ComponentName({1}) = "error";
    var.ComponentName({2}) = "maxBlockError";
    
    cmf::Vec3<> x(0.01, 0.01, 0.01);
    // domain.Blocks()->RefineAt(x, 7);
    FillArr(var);
    
    // var.Exchange();
    var.ExportFile("output", "test");
    
    domain.GetPartition()->OutputPartitionToVtk("partition.vtk");
    
    return 0;
}