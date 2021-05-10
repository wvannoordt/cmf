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
        int ijk[3] = {0};
        for (cell_t k = arLb.kmin; k < arLb.kmax; k++)
        {
            ijk[2] = k;
            for (cell_t j = arLb.jmin; j < arLb.jmax; j++)
            {
                ijk[1] = j;
                for (cell_t i = arLb.imin; i < arLb.imax; i++)
                {
                    ijk[0] = i;
                    double xyz[3] = {0.0};
                    for (int d = 0; d < CMF_DIM; d++)
                    {
                        xyz[d] = info.blockBounds[2*d] + (0.5 + (double)ijk[d])*info.dx[d];
                    }
                    arLb(0, i, j, k) = sin(xyz[0])*cos(xyz[1]-0.4) + cos(2.0*xyz[0])+sin(xyz[2]);
                    arLb(1, i, j, k) = sin(2*xyz[0])*cos(xyz[1]-0.4) + cos(2.0*xyz[0])+sin(xyz[2]);
                    arLb(2, i, j, k) = cos(xyz[0])*sin(xyz[1]-0.4) + sin(2.0*xyz[0])+sin(xyz[2]);
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    std::string inFile = strformat("input{}D.ptl", CMF_DIM);
    cmf::ReadInput(inFile);
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
    domain.Blocks()->RefineNodes(nodes, 1);
    std::vector<decltype(node)> nodes2;
    auto node2 = domain.Blocks()->GetNodeAt(coords);
    nodes2.push_back(node2);
    domain.Blocks()->RefineNodes(nodes2, 2);
    
    auto& var = domain.DefineVariable("preData", cmf::CmfArrayType::CmfDouble, {3});
    
    var.ComponentName({0}) = "array0";
    var.ComponentName({1}) = "array1";
    var.ComponentName({2}) = "array2";
    
    FillArGhost(var, -1.0);
    FillAr(var);
    
    var.ExportFile("output", "test");
    
    return 0;
}