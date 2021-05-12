#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
using cmf::print;
using cmf::strformat;
using cmf::strunformat;
using cmf::cell_t;
#define R0 25.0
#define d_abs(a) (((a)>0)?(a):(-(a)))
double func(double x, double y, double z)
{
    double r = sqrt(x*x + y*y + z*z);
    return exp(-(r - R0)*(r - R0));
}

double func_x(double x, double y, double z)
{
    double r = sqrt(x*x + y*y + z*z);
    return -2*x*(r-R0)*exp(-(r - R0)*(r - R0))/r;
}

double func_y(double x, double y, double z)
{
    double r = sqrt(x*x + y*y + z*z);
    return -2*y*(r-R0)*exp(-(r - R0)*(r - R0))/r;
}

double func_z(double x, double y, double z)
{
    double r = sqrt(x*x + y*y + z*z);
    return -2*z*(r-R0)*exp(-(r - 25.0)*(r - 25.0))/r;
}

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
                    arLb(3, i, j, k) = val;
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
                    arLb(0, i, j, k) = func(xyz[0], xyz[1], xyz[2]);
                    arLb(1, i, j, k) = func_x(xyz[0], xyz[1], xyz[2]);
                    arLb(2, i, j, k) = func_y(xyz[0], xyz[1], xyz[2]);
                    arLb(3, i, j, k) = func_z(xyz[0], xyz[1], xyz[2]);
                }
            }
        }
    }
}

bool refineArr(cmf::CartesianMeshArray& ar)
{
    int maxRefine = 1;
    bool done = false;
    int count = 0;
    while(!done && (count++ < maxRefine))
    {
        std::map<cmf::RefinementTreeNode*, char> refMap;
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
                        double dx = info.dx[0];
                        double dy = info.dx[1];
                        double dz = CMF_IS3D*info.dx[CMF_DIM-1] + (1-CMF_IS3D)*1.0;
                        double ff = func(xyz[0], xyz[1], xyz[2]);
                        double gx = func_x(xyz[0], xyz[1], xyz[2]);
                        double gy = func_y(xyz[0], xyz[1], xyz[2]);
                        double gz = func_z(xyz[0], xyz[1], xyz[2]);
                        
                        gx = d_abs(gx);
                        gy = d_abs(gy);
                        gz = d_abs(gz);
                        
                        double refineVal = 0.2;
                        char refX = (gx > refineVal)?1:0;
                        char refY = (gy > refineVal)?2:0;
                        char refZ = (gz > refineVal)?4:0;
                        if (!CMF_IS3D) refZ = 0;
                        char result = refX;
                        result = result | refY;
                        result = result | refZ;
                        // result = 0;
                        // if (ff>0.1) result = 3;
                        if (result>0)
                        {
                            if (refMap.find(lb) != refMap.end())
                            {
                                char oldResult = refMap[lb];
                                refMap[lb] = result|oldResult;
                            }
                            else
                            {
                                refMap.insert({lb, result});
                            }
                        }
                    }
                }
            }
        }
        std::vector<cmf::RefinementTreeNode*> nodes;
        std::vector<char> refs;
        for (auto& pr:refMap)
        {
            auto& np = pr.first;
            double* bds = np->GetBlockBounds();
            nodes.push_back(pr.first);
            refs.push_back(pr.second);
        }
        done = (nodes.size()==0);
        if (!done)
        {
            ar.Mesh()->Blocks()->RefineNodes(nodes, refs);
        }
    }
    return done;
}

void SillyRefine(cmf::CartesianMeshArray& ar)
{
    std::vector<cmf::RefinementTreeNode*> nodes;
    char refType = 3;
    double coords[CMF_DIM]={0.3};
    nodes.push_back(ar.Mesh()->Blocks()->GetNodeAt(coords));
    ar.Mesh()->Blocks()->RefineNodes(nodes, refType);
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
    auto& var = domain.DefineVariable("preData", cmf::CmfArrayType::CmfDouble, {4});
    // refineArr(var);
    // refineArr(var);
    // refineArr(var);
    SillyRefine(var);
    
    var.ComponentName({0}) = "f";
    var.ComponentName({1}) = "df_dx";
    var.ComponentName({2}) = "df_dy";
    var.ComponentName({3}) = "df_dz";
    
    FillArGhost(var, -1.0);
    FillAr(var);
    
    var.ExportFile("output", "test");
    
    return 0;
}