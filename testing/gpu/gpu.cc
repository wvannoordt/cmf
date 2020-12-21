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
    
    double* sampleCoords;
    bool doRefinement;
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CreateParallelContext(&argc, &argv);
    PTL::PropertySection User = cmf::mainInput["User"];
    User["sampleCoords"].MapTo(&sampleCoords) = new PTL::Variables::PTLStaticDoubleArray(cmf::GetDim(), "Sampling coordinates", [](int i){return 0.01;});
    User["doRefinement"].MapTo(&doRefinement) = new PTL::Variables::PTLBoolean(false, "Perform refinemet");
    User.StrictParse();
    
    cmf::CartesianMeshParallelPartitionInfo domainPartition(cmf::mainInput["Domain"]["Partition"]); // note that if the order of this and the declaration
    // of inputInfo is switched, PTL will be sad :(
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    cmf::CartesianMeshParallelPartition* partition = domain.CreateParallelPartition(domainPartition);
    double coords[3];
    //coords[0] = 0.2 + 0.4*(1-cmf::globalGroup.Rank()); // <-- will cause the later assertion to fail.
    coords[0] = 0.5;
    coords[1] = 0.5;
    coords[2] = 0.5;
    double radius = 0.1;
    cmf::AxisAlignedLongCylinder cyl(coords, radius, 0);
    domain.Blocks()->SetRefineLimitCriterion([](cmf::RefinementTreeNode* n){return (n->GetLevel() > LEVEL);});
    int z = 1;
    if (doRefinement)
    {
        for (cmf::BlockIterator lb(domain.Blocks(), cmf::BlockFilters::Terminal); lb.HasNext(); lb++)
        {
            double* bounds = lb.Node()->GetBlockBounds();
            bool intersect = cyl.BoxIntersectsBoundary(bounds);
            if (intersect)
            {
                lb.Node()->Refine(z++);
                if (z==8) z=1;
            }
        }
        domain.Blocks()->PostRefinementCallbacks();
    }
    return 0;
}