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
    bool outputFile, doRefinement;
    double* sampleCoords;
    std::string outputFileName;
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CreateParallelContext(&argc, &argv);
    PTL::PropertySection user = cmf::mainInput["User"];
    user["outputFile"].MapTo(&outputFile) = new PTL::PTLBoolean(false, "Output the partition file");
    user["sampleCoords"].MapTo(&sampleCoords) = new PTL::PTLStaticDoubleArray(3, "Refinement coordinates", [](int i){return 0.1;});
    user["doRefinement"].MapTo(&doRefinement) = new PTL::PTLBoolean(false, "Refine the mesh at the sample coordinates");
    user["outputFileName"].MapTo(&outputFileName) = new PTL::PTLString("data.vtk", "Name of the output file");
    user.StrictParse();
    if (cmf::HasGpuSupport())
    {
        //Test to see if we can allocate a gpu buffer
        double* devPtr = (double*)Cmf_GpuAlloc(10*sizeof(double), 0);
        Cmf_GpuFree(devPtr);
    }
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    cmf::RefinementTreeNode* node = domain.Blocks()->GetNodeAt(sampleCoords);
    domain.CreateCoordinateVariable(0);
    if (doRefinement)
    {
        if (cmf::globalGroup.IsRoot())
        {
            std::cout << "Refine at " << sampleCoords[0] << ", " << sampleCoords[1] << ", " << sampleCoords[2] << std::endl;
        }
        node->Refine(7);
        domain.Blocks()->PostRefinementCallbacks();
    }
    
    int minTimeStep = 0;
    int maxTimeStep = 50;
    for (int i = minTimeStep; i <= maxTimeStep; i++)
    {
        if (cmf::globalGroup.IsRoot())
        {
            std::cout << "Step " << i << std::endl;
            //Do exchanges
        }
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