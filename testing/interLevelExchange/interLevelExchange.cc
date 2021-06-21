#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>

using cmf::print;
using cmf::strformat;
using cmf::strunformat;
using cmf::cell_t;
using cmf::ZFill;

#define DMAX(a,b) ((a)>(b)?(a):(b))
#define DABS(a) (((a)>0)?(a):(-(a)))

#define PI 3.141592655359

const double ghostJunkValue = -10.0;

double fxyz(double x, double y, double z)
{
    double alpha = 2.0;
    return cos(4*PI*alpha*y)*sin(PI*alpha*z) + sin(2*PI*alpha*y)*cos(3*PI*alpha*x) + cos(2*PI*alpha*z)*cos(2*PI*alpha*x);
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
    std::vector<cmf::RefinementTreeNode*> nodes2;
    std::vector<char> refs2;
    
    nodes2.push_back(ar.Mesh()->Blocks()->GetNodeAt(0.2, 0.6, 0.0));
    refs2.push_back(1);
    nodes2.push_back(ar.Mesh()->Blocks()->GetNodeAt(1.2, 1.6, 0.0));
    refs2.push_back(2);
    nodes2.push_back(ar.Mesh()->Blocks()->GetNodeAt(0.2, 0.9, 0.0));
    refs2.push_back(3);
    
    ar.Mesh()->Blocks()->RefineNodes(nodes2, refs2);
    nodes2.clear();
    refs2.clear();
    
    nodes2.push_back(ar.Mesh()->Blocks()->GetNodeAt(0.2, 0.6, 0.0));
    refs2.push_back(1);
    nodes2.push_back(ar.Mesh()->Blocks()->GetNodeAt(1.2, 1.6, 0.0));
    refs2.push_back(2);
    nodes2.push_back(ar.Mesh()->Blocks()->GetNodeAt(0.2, 0.9, 0.0));
    refs2.push_back(3);
    
    ar.Mesh()->Blocks()->RefineNodes(nodes2, refs2);
    nodes2.clear();
    refs2.clear();
    
    nodes2.push_back(ar.Mesh()->Blocks()->GetNodeAt(0.2, 0.6, 0.0));
    refs2.push_back(1);
    nodes2.push_back(ar.Mesh()->Blocks()->GetNodeAt(1.2, 1.6, 0.0));
    refs2.push_back(2);
    nodes2.push_back(ar.Mesh()->Blocks()->GetNodeAt(0.2, 0.9, 0.0));
    refs2.push_back(3);
    
    ar.Mesh()->Blocks()->RefineNodes(nodes2, refs2);
    nodes2.clear();
    refs2.clear();
    
    nodes2.push_back(ar.Mesh()->Blocks()->GetNodeAt(1.4, 1.3, 0.0));
    refs2.push_back(3);
    
    ar.Mesh()->Blocks()->RefineNodes(nodes2, refs2);
    nodes2.clear();
    refs2.clear();
    
    // nodes2.push_back(ar.Mesh()->Blocks()->GetNodeAt(1.4, 1.3, 0.0));
    // refs2.push_back(1);
    // 
    // ar.Mesh()->Blocks()->RefineNodes(nodes2, refs2);
    // nodes2.clear();
    // refs2.clear();
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
                        
                        if (DABS(arLb(i, j, k) - ghostJunkValue) < 1e-8)
                        {
                            print("BAD GHOST CELL VALUE");
                            print(i, j, k);
                            print(lb->GetBlockCenter());
                            abort();
                        }
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

bool Test1DInterpolationOperator(void)
{
    std::vector<double> logError;
    std::vector<double> logDx;
    int order = 5;
    for (int i = 1; i < 6; i++)
    {
        cmf::InterpolationOperator1D interp;
        interp.order = order;
        interp.SetSize(18);
        double x0 = 0.0;
        double x1 = 0.917/i;
        double xs = 0.1;
        double dxx = (x1-x0)/(interp.size-1);
        logDx.push_back(log10(dxx));
        for (int j = 0; j < interp.size; j++)
        {
            interp.coords[j] = x0 + j*dxx;
            interp.data[j] = sin(interp.coords[j])+2.0*cos(2.0*interp.coords[j]);
        }
        double xana = sin(xs)+2.0*cos(2.0*xs);
        int imin = interp.FindMinStencilIndex(xs);
        double result = 0.0;
        for (int i = 0; i < interp.order; i++)
        {
            double coeff = interp.GetCoefficientAtPoint(imin, i, xs);
            double data = interp.data[imin+i];
            result += coeff*data;
        }
        logError.push_back(log10(abs(xana-result)));
    }
    
    double sum = 0.0;
    for (int i = 0; i < logError.size()-1; i++)
    {
        sum += (logError[i+1] - logError[i])/(logDx[i+1] - logDx[i]);
    }
    sum /= (logError.size()-1);
    return sum > (order-0.7);
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
    
    var.ComponentName() = "fxyz";
    
    FillArGhost(var, ghostJunkValue);
    FillAr(var);
    
    auto interlevels = var.GetExchangePattern()->GetTransactionsByType<cmf::CartesianInterLevelBlockTransaction<double>>();
    for (int i = 0; i < interlevels.size(); i++)
    {
        auto exch = interlevels[i];
        cmf::DebugPointCloud pc1, pc2;
        exch->GetSendInfo().GetExchangeRegionAsPointCloud(pc1);
        exch->GetRecvInfo().GetExchangeRegionAsPointCloud(pc2);
        std::string sendFilename = strformat("output/send_{}.vtk", ZFill(i, 7));
        std::string recvFilename = strformat("output/recv_{}.vtk", ZFill(i, 7));
        pc1.WriteVtk(sendFilename);
        pc2.WriteVtk(recvFilename);
    }
    
    var.Exchange();
    
    
    
    var.ExportFile("output", "test");
    
    double errInf = 0.0;
    double errL2  = 0.0;
    EvalErr(var, errL2, errInf);
    
    if (!Test1DInterpolationOperator())
    {
        print("1D INTERPOLATION OPERATOR TEST FAILED");
        return 1;
    }
    return 0;
}