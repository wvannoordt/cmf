#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
using cmf::print;
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
        cmf::CmfDataBase outputDatabase("output");
        auto& var = domain.DefineVariable("data");
        outputDatabase << domain;
        outputDatabase << var;
        
        outputDatabase.Write(dataFileTitle);
    }

    {
        
        cmf::CartesianMesh domain(inputInfo);
        // cmf::CmfDataBase inputDatabase("output");
        // inputDatabase.Read("testDatabase");
        // outputDatabase >> domain;
        // outputDatabase >> var;
        // 
        // cmf::CartesianMesh* domain;
        
        
    }
    
    return 0;
}