#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
using cmf::print;
using cmf::strformat;
using cmf::strunformat;
using cmf::cell_t;

void FillArGhost(cmf::CartesianMeshArray& ar, double val)
{
    for (auto lb: ar)
    {
        cmf::BlockArray<double> arLb = ar[lb];
        cmf::BlockInfo info = ar.Mesh()->GetBlockInfo(lb);
        for (cell_t k = arLb.kmin-arLb.exchangeK; k < arLb.kmax+arLb.exchangeK; k++)
        {
            for (cell_t j = arLb.jmin-arLb.exchangeJ; j < arLb.jmax+arLb.exchangeJ; j++)
            {
                for (cell_t i = arLb.imin-arLb.exchangeI; i < arLb.imax+arLb.exchangeI; i++)
                {
                    arLb(i, j, k) = val;
                }
            }
        }
    }
}

void FillAr(cmf::CartesianMeshArray& ar, double val)
{
    for (auto lb: ar)
    {
        cmf::BlockArray<double> arLb = ar[lb];
        cmf::BlockInfo info = ar.Mesh()->GetBlockInfo(lb);
        for (cell_t k = arLb.kmin; k < arLb.kmax; k++)
        {
            for (cell_t j = arLb.jmin; j < arLb.jmax; j++)
            {
                for (cell_t i = arLb.imin; i < arLb.imax; i++)
                {
                    arLb(i, j, k) = val;
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
    double coords[3];
    coords[0] = 0.3;
    coords[1] = 0.3;
    coords[2] = 0.3;
    auto node = domain.Blocks()->GetNodeAt(coords);
    std::vector<decltype(node)> nodes;
    nodes.push_back(node);
    // domain.Blocks()->RefineNodes(nodes, 7);
    
    auto& var = domain.DefineVariable("preData");
    FillArGhost(var, -1.0);
    FillAr(var, 0.9);
    
    var.ExportFile("output", "test");
    
    return 0;
}