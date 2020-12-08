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
    
    PropTreeLib::PropertySection User = cmf::mainInput["User"];
    User["sampleCoords"].MapTo(&sampleCoords) = new PropTreeLib::Variables::PTLStaticDoubleArray(cmf::GetDim(), "Sampling coordinates", [](int i){return 0.01;});
    User["doRefinement"].MapTo(&doRefinement) = new PropTreeLib::Variables::PTLBoolean(false, "Perform refinemet");
    User.StrictParse();
    
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    //The partitioning object should be created here, or after the refinement is complete.
    double coords[3];
    //coords[0] = 0.2 + 0.4*(1-cmf::globalGroup.Rank()); // <-- will cause the later assertion to fail.
    coords[0] = 0.52;
    coords[1] = 0.58;
    coords[2] = 0.51231;
    double radius = 0.1;
    cmf::AxisAlignedLongCylinder cyl(coords, radius, 2);
    domain.Blocks()->SetRefineLimitCriterion([](cmf::RefinementTreeNode* n){return (n->GetLevel() > LEVEL);});
    if (doRefinement)
    {
        for (cmf::BlockIterator lb(domain.Blocks(), cmf::BlockFilters::Terminal); lb.HasNext(); lb++)
        {
            double* bounds = lb.Node()->GetBlockBounds();
            bool intersect = cyl.BoxIntersectsBoundary(bounds);
            if (intersect)
            {
                lb.Node()->Refine(7);
            }
        }
    }
    domain.AssertSynchronizeBlocks();
    cmf::CartesianMeshArray x = domain.CreateCoordinateVariable(0);
    return 0;
}