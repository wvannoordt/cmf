#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
using cmf::print;
using cmf::strformat;
using cmf::strunformat;
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
        
        auto& var = domain.DefineVariable("data");
        
        domain.Blocks()->OutputDebugVtk("output/beforeWrite.vtk");
        
        cmf::CmfDataBase outputDatabase("output");
        outputDatabase["mesh"] << domain;
        outputDatabase["data"] << var;
        
        outputDatabase.Write(dataFileTitle);
        
        cmf::CartesianMesh domain2(inputInfo);
        
        cmf::CmfDataBase inputDataBase("output");
        
        inputDataBase["mesh"] >> domain2;
        
        inputDataBase.Read(dataFileTitle);
        
        domain2.Blocks()->OutputDebugVtk("output/afterWrite.vtk");
    }
    
    return 0;
}