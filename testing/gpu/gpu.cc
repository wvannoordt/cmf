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

void FillArr(cmf::CartesianMeshArray& arr)
{
    for (auto lb: arr)
    {
        cmf::BlockArray<double, 1> arLb = arr[lb];
        FillBlockGpu(arLb);
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
    
    var.ComponentName({0}) = "i";
    var.ComponentName({1}) = "j";
    var.ComponentName({2}) = "k";
    
    cmf::Vec3<> x(0.01, 0.01, 0.01);
    domain.Blocks()->RefineAt(x, 7);
    
    FillArr(var);
    
    // var.Exchange();
    var.ExportFile("output", "test");
    
    return 0;
}