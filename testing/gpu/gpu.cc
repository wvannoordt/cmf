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
    user["outputFile"].MapTo(&outputFile) = new PTL::Variables::PTLBoolean(false, "Output the partition file");
    user["sampleCoords"].MapTo(&sampleCoords) = new PTL::Variables::PTLStaticDoubleArray(3, "Refinement coordinates", [](int i){return 0.1;});
    user["doRefinement"].MapTo(&doRefinement) = new PTL::Variables::PTLBoolean(false, "Refine the mesh at the sample coordinates");
    user["outputFileName"].MapTo(&outputFileName) = new PTL::Variables::PTLString("data.vtk", "Name of the output file");
    user.StrictParse();
    
    cmf::CartesianMeshParallelPartitionInfo domainPartition(cmf::mainInput["Domain"]["Partition"]); // note that if the order of this and the declaration
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    cmf::CartesianMeshParallelPartition* partition = domain.CreateParallelPartition(domainPartition);
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
    if (outputFile)
    {
        if (cmf::globalGroup.IsRoot())
        {
            std::cout << "Output " << outputFileName << std::endl;
        }
    }
    return 0;
}