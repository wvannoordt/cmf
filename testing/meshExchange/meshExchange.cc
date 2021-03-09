#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>

void FillArray(cmf::CartesianMeshArray& ar, double value, bool doGuardFilling)
{
    
}

int main(int argc, char** argv)
{
    EXIT_WARN_IF_DIM_NOT(3);
    
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    
    cmf::CreateParallelContext(&argc, &argv);
    
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    auto& var = domain.DefineVariable("var", sizeof(double));
    
    
    
    return 0;
}