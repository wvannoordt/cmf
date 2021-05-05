#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
using cmf::print;
using cmf::strformat;
using cmf::strunformat;
using cmf::cell_t;

void FillAr(cmf::CartesianMeshArray& ar, double val)
{
    int y = -1;
    for (auto lb: ar)
    {
        y++;
        cmf::BlockArray<double> arLb = ar[lb];
        cmf::BlockInfo info = ar.Mesh()->GetBlockInfo(lb);
        for (cell_t k = arLb.kmin-arLb.exchangeK; k < arLb.kmax+arLb.exchangeK; k++)
        {
            for (cell_t j = arLb.jmin-arLb.exchangeJ; j < arLb.jmax+arLb.exchangeJ; j++)
            {
                for (cell_t i = arLb.imin-arLb.exchangeI; i < arLb.imax+arLb.exchangeI; i++)
                {
                    arLb(i, j, k) = val+y;
                }
            }
        }
    }
}

bool CheckArr(cmf::CartesianMeshArray& ar, double val)
{
    double localErrorSq = 0.0;
    int y = -1;
    for (auto lb: ar)
    {
        y++;
        cmf::BlockArray<double> arLb = ar[lb];
        cmf::BlockInfo info = ar.Mesh()->GetBlockInfo(lb);
        for (cell_t k = arLb.kmin-arLb.exchangeK; k < arLb.kmax+arLb.exchangeK; k++)
        {
            for (cell_t j = arLb.jmin-arLb.exchangeJ; j < arLb.jmax+arLb.exchangeJ; j++)
            {
                for (cell_t i = arLb.imin-arLb.exchangeI; i < arLb.imax+arLb.exchangeI; i++)
                {
                    localErrorSq += (arLb(i, j, k) - (val+y))*(arLb(i, j, k) - (val+y));
                }
            }
        }
    }
    double globErrorSq = cmf::globalGroup.Sum(localErrorSq);
    double globalError = sqrt(globErrorSq);
    return globalError < 1e-6;
}

int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    EXIT_WARN_IF_DIM_NOT(3);
    
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CreateParallelContext(&argc, &argv);
    
    PTL::PropertySection& user = cmf::mainInput["User"];

    std::string dataFileTitle;
    user["dataFileTitle"].MapTo(&dataFileTitle) = new PTL::PTLString("cmfData", "The name of the data file to be output");

    user.StrictParse();
    
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    
    bool success = false;
    
    {
        cmf::CartesianMesh domain(inputInfo);        
        double coords[3];
        coords[0] = 0.3;
        coords[1] = 0.3;
        coords[2] = 0.3;
        auto node = domain.Blocks()->GetNodeAt(coords);
        std::vector<decltype(node)> nodes;
        nodes.push_back(node);
        domain.Blocks()->RefineNodes(nodes, 7);
        
        auto& var = domain.DefineVariable("preData");
        FillAr(var, 0.9);
        
        domain.Blocks()->OutputDebugVtk("output/beforeWrite.vtk");
        
        cmf::CmfDataBase outputDatabase("output");
        outputDatabase["mesh"] << domain;
        outputDatabase["preData"] << var;
        
        outputDatabase.Write(dataFileTitle);
        
        cmf::CartesianMesh domain2(inputInfo);
        auto& var2 = domain2.DefineVariable("postData");
        
        cmf::CmfDataBase inputDataBase("output");
        
        inputDataBase["mesh"] >> domain2;
        inputDataBase["preData"] >> var2;
        
        
        
        inputDataBase.Read(dataFileTitle);
        
        success = CheckArr(var, 0.9);
        
        domain2.Blocks()->OutputDebugVtk("output/afterWrite.vtk");
    }
    if (!success)
    {
        if (cmf::globalGroup.IsRoot())
        {
            print("Error in test case: failed database input/output check!", __FILE__);
        }
    }
    return success?0:1;
}