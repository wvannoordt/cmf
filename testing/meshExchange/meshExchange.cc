#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>

void FillArray(cmf::CartesianMeshArray& ar, double value, bool doGuardFilling)
{
    for (auto lb: ar)
    {
        cmf::BlockArray<double> block = ar[lb];
        for (int k = block.kmin; k < block.kmax; k++)
        {
            for (int j = block.jmin; j < block.jmax; j++)
            {
                for (int i = block.imin; i < block.imax; i++)
                {
                    block(i, j, k) = 1.0;
                }
            }
        }
    }
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