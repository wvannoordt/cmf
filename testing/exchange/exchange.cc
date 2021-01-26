#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
#define LEVEL 2
int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    EXIT_WARN_IF_DIM_NOT(3);
    bool outputFile;
    double* sampleCoords;
    int minTimeStep;
    int maxTimeStep;
    std::string outputFileName;
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CreateParallelContext(&argc, &argv);
    PTL::PropertySection user = cmf::mainInput["User"];
    user["outputFile"].MapTo(&outputFile) = new PTL::PTLBoolean(false, "Output the partition file");
    user["sampleCoords"].MapTo(&sampleCoords) = new PTL::PTLStaticDoubleArray(3, "Refinement coordinates", [](int i){return 0.1;});
    user["outputFileName"].MapTo(&outputFileName) = new PTL::PTLString("data.vtk", "Name of the output file");
    user["minTimeStep"].MapTo(&minTimeStep) = new PTL::PTLInteger(0, "Minimum time step");
    user["maxTimeStep"].MapTo(&maxTimeStep) = new PTL::PTLInteger(1, "Maximum time step");
    user.StrictParse();
    
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    cmf::RefinementTreeNode* node = domain.Blocks()->GetNodeAt(sampleCoords);
    
    domain.CreateCoordinateVariable(0);
    
    for (int i = minTimeStep; i <= maxTimeStep; i++)
    {
        if (cmf::globalGroup.IsRoot())
        {
            std::cout << "Step " << i << std::endl;
        }
        //Do exchanges
        domain["x"].Exchange();
    }
    
    if (outputFile)
    {
        if (cmf::globalGroup.IsRoot())
        {
            std::cout << "Output " << outputFileName << std::endl;
        }
    }
    return 0;
}