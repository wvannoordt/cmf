#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>

void FillArray(cmf::CartesianMeshArray& ar, double value, bool doGuardFilling = false)
{
    int ng = doGuardFilling?1:0;
    for (auto lb: ar)
    {
        cmf::BlockArray<double> block = ar[lb];
        for (int k = block.kmin - ng*block.exchangeK; k < block.kmax + ng*block.exchangeK; k++)
        {
            for (int j = block.jmin - ng*block.exchangeJ; j < block.jmax + ng*block.exchangeJ; j++)
            {
                for (int i = block.imin - ng*block.exchangeI; i < block.imax + ng*block.exchangeI; i++)
                {
                    block(i, j, k) = value;
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    EXIT_WARN_IF_DIM_NOT(3);
    
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CreateParallelContext(&argc, &argv);
    
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    auto& var = domain.DefineVariable("data", sizeof(double));
    FillArray(var, -1.0, true);
    FillArray(var, (double)(cmf::globalGroup.Rank()), false);
    if (cmf::globalGroup.Size()<2)
    {
        cmf::SerialCartesianVtk svtk(domain, "output/data.vtk");
        svtk << var;
        svtk.Write();
    }
    
    return 0;
}