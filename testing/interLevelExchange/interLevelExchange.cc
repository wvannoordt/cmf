#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>

using cmf::print;
using cmf::strformat;
using cmf::strunformat;
using cmf::cell_t;

#define DMAX(a,b) ((a)>(b)?(a):(b))
#define DABS(a) (((a)>0)?(a):(-(a)))

#define PI 3.141592655359

double fxyz(double x, double y, double z)
{
    return cos(4*PI*y)*sin(PI*z) + sin(2*PI*y)*cos(3*PI*x) + cos(2*PI*z)*cos(2*PI*x);
}

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

void FillAr(cmf::CartesianMeshArray& ar)
{
    for (auto lb: ar)
    {
        cmf::BlockArray<double> arLb = ar[lb];
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
                    arLb(i, j, k) = fxyz(xyz[0], xyz[1], xyz[2]);
                }
            }
        }
    }
}

void SillyRefine(cmf::CartesianMeshArray& ar)
{
    std::vector<cmf::RefinementTreeNode*> nodes;
    char refType = 3;
    double coords[CMF_DIM]={0.1};
    nodes.push_back(ar.Mesh()->Blocks()->GetNodeAt(coords));
    ar.Mesh()->Blocks()->RefineNodes(nodes, refType);
}

void EvalErr(cmf::CartesianMeshArray& ar, double& l2Err, double& linfErr)
{
    l2Err = 0.0;
    linfErr = 0.0;
    double l2ErrLocal = 0.0;
    double linfErrLocal = 0.0;
    int dof = 0;
    for (auto lb: ar)
    {
        cmf::BlockArray<double> arLb = ar[lb];
        cmf::BlockInfo info = ar.Mesh()->GetBlockInfo(lb);
        int ijk[3] = {0};
        auto iGhost = [&](int ii, int jj, int kk) -> bool 
        {
            return (ii < arLb.imin)||(ii >= arLb.imax)||(kk < arLb.kmin)||(kk >= arLb.kmax)||(jj < arLb.jmin)||(jj >= arLb.jmax);
        };
        for (cell_t k = arLb.kmin - arLb.exchangeK; k < arLb.kmax + arLb.exchangeK; k++)
        {
            ijk[2] = k;
            for (cell_t j = arLb.jmin - arLb.exchangeJ; j < arLb.jmax + arLb.exchangeJ; j++)
            {
                ijk[1] = j;
                for (cell_t i = arLb.imin - arLb.exchangeI; i < arLb.imax + arLb.exchangeI; i++)
                {
                    ijk[0] = i;
                    double xyz[3] = {0.0};
                    for (int d = 0; d < CMF_DIM; d++)
                    {
                        xyz[d] = info.blockBounds[2*d] + (0.5 + (double)ijk[d])*info.dx[d];
                    }
                    if (iGhost(i, j, k))
                    {
                        dof++;
                        double errLoc = (arLb(i, j, k) - fxyz(xyz[0], xyz[1], xyz[2]));
                        l2ErrLocal += errLoc*errLoc;
                        linfErrLocal = DMAX(linfErrLocal, DABS(errLoc));
                    }
                }
            }
        }
    }
    auto& group = *(ar.Mesh()->GetGroup());
    int totaldof = group.Sum(dof);
    l2Err = sqrt(group.Sum(l2ErrLocal))/totaldof;
    linfErr = group.Max(linfErrLocal);
}

int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    std::string inFile = "input.ptl";
    cmf::ReadInput(inFile);
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CreateParallelContext(&argc, &argv);

    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);        
    auto& var = domain.DefineVariable("preData", cmf::CmfArrayType::CmfDouble);
    SillyRefine(var);
    
    var.ComponentName({}) = "fxyz";
    
    FillArGhost(var, -10.0);
    FillAr(var);
    
    var.Exchange();
    
    double errInf = 0.0;
    double errL2  = 0.0;
    EvalErr(var, errL2, errInf);
    
    var.ExportFile("output", "test");
    
    return 0;
}