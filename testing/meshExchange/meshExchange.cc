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
        for (cmf::cell_t k = block.kmin - ng*block.exchangeK; k < block.kmax + ng*block.exchangeK; k++)
        {
            for (cmf::cell_t j = block.jmin - ng*block.exchangeJ; j < block.jmax + ng*block.exchangeJ; j++)
            {
                for (cmf::cell_t i = block.imin - ng*block.exchangeI; i < block.imax + ng*block.exchangeI; i++)
                {
                    block(i, j, k) = value;
                }
            }
        }
    }
}

void OutputIndividualBlocks(cmf::CartesianMeshArray& ar, bool isPostExchange = false)
{
    int p = 0;
    for (auto lb: ar)
    {
        auto info = ar.GetBlockInfo(lb);
        cmf::BlockArray<double> block = ar[lb];
        std::string filename;
        if (isPostExchange)
        {
            filename = "output/blockPostExchange_proc" + std::to_string(cmf::globalGroup.Rank()) + "_block" + std::to_string(p) + ".vtk";
        }
        else
        {
            filename = "output/blockPreExchange_proc" + std::to_string(cmf::globalGroup.Rank()) + "_block" + std::to_string(p) + ".vtk";
        }
        std::cout << "output " << filename << std::endl;
        std::ofstream myfile;
        myfile.open(filename.c_str());
        myfile << "# vtk DataFile Version 3.0" << std::endl;
        myfile << "vtk output" << std::endl;
        myfile << "ASCII" << std::endl;
        myfile << "DATASET STRUCTURED_POINTS" << std::endl;
        myfile << "DIMENSIONS ";
        int ni = block.imax - block.imin + 2*block.exchangeI;
        int nj = block.jmax - block.jmin + 2*block.exchangeJ;
        int nk = block.kmax - block.kmin + 2*block.exchangeK;
        myfile << (ni+1) << " ";
        myfile << (nj+1) << " ";
        myfile << (nk+1) << std::endl;
        
        double origin[3] = {0.0};
        double spacing[3] = {1.0};
        for (int i = 0; i < CMF_DIM; i++) origin[i] = info.blockBounds[2*i];
        for (int i = 0; i < CMF_DIM; i++) spacing[i] = info.dx[i];
        
        myfile << "ORIGIN "  << origin[0]  << " " << origin[1]  << " " << origin[2]  << std::endl;
        myfile << "SPACING " << spacing[0] << " " << spacing[1] << " " << spacing[2] << std::endl;
        myfile << "CELL_DATA " << ni*nj*nk << std::endl;
        myfile << "SCALARS Rank double"  << std::endl;
        myfile << "LOOKUP_TABLE default" << std::endl;
        for (cmf::cell_t k = block.kmin - block.exchangeK; k < block.kmax + block.exchangeK; k++)
        {
            for (cmf::cell_t j = block.jmin - block.exchangeJ; j < block.jmax + block.exchangeJ; j++)
            {
                for (cmf::cell_t i = block.imin - block.exchangeI; i < block.imax + block.exchangeI; i++)
                {
                    myfile << block(i, j, k) << std::endl;
                }
            }
        }
        myfile.close();
        p++;
    }
}

int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    EXIT_WARN_IF_DIM_NOT(3);
    
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CreateParallelContext(&argc, &argv);
    
    PTL::PropertySection& user = cmf::mainInput["User"];
    
    bool doOutput;
    user["doOutput"].MapTo(&doOutput) = new PTL::PTLBoolean(false, "Output individual block files with guard cells");
    
    user.StrictParse();
    
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    
    auto& var = domain.DefineVariable("data", sizeof(double));
    FillArray(var, -1.0, true);
    FillArray(var, (double)(cmf::globalGroup.Rank()), false);
    domain.GetPartition()->OutputPartitionToVtk("output/partition.vtk");
    if (doOutput) OutputIndividualBlocks(var);
    var.Exchange();
    if (doOutput) OutputIndividualBlocks(var, true);
    if (cmf::globalGroup.Size()<2)
    {
       cmf::SerialCartesianVtk svtk(domain, "output/data.vtk");
       svtk << var;
       svtk.Write();
    }
    return 0;
}