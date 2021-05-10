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
        cmf::BlockArray<double, 1> arLb = ar[lb];
        cmf::BlockInfo info = ar.Mesh()->GetBlockInfo(lb);
        for (cell_t k = arLb.kmin-arLb.exchangeK; k < arLb.kmax+arLb.exchangeK; k++)
        {
            for (cell_t j = arLb.jmin-arLb.exchangeJ; j < arLb.jmax+arLb.exchangeJ; j++)
            {
                for (cell_t i = arLb.imin-arLb.exchangeI; i < arLb.imax+arLb.exchangeI; i++)
                {
                    arLb(0, i, j, k) = val;
                    arLb(1, i, j, k) = val;
                    arLb(2, i, j, k) = val;
                }
            }
        }
    }
}

void FillAr(cmf::CartesianMeshArray& ar)
{
    for (auto lb: ar)
    {
        cmf::BlockArray<double, 1> arLb = ar[lb];
        cmf::BlockInfo info = ar.Mesh()->GetBlockInfo(lb);
        for (cell_t k = arLb.kmin; k < arLb.kmax; k++)
        {
            for (cell_t j = arLb.jmin; j < arLb.jmax; j++)
            {
                for (cell_t i = arLb.imin; i < arLb.imax; i++)
                {
                    double x = info.blockBounds[0] + (0.5 + (double)i)*info.dx[0];
                    double y = info.blockBounds[2] + (0.5 + (double)j)*info.dx[1];
                    double z = info.blockBounds[4] + (0.5 + (double)k)*info.dx[2];
                    arLb(0, i, j, k) = sin(x)*cos(y-0.4) + cos(2.0*x)+sin(z);
                    arLb(1, i, j, k) = sin(2*x)*cos(y-0.4) + cos(2.0*x)+sin(z);
                    arLb(2, i, j, k) = cos(x)*sin(y-0.4) + sin(2.0*x)+sin(z);
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
    domain.Blocks()->RefineNodes(nodes, 3);
    
    auto& var = domain.DefineVariable("preData", cmf::CmfArrayType::CmfDouble, {3});
    FillArGhost(var, -1.0);
    FillAr(var);
    
    var.ExportFile("output", "test");
    
    return 0;
}