#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    EXIT_WARN_IF_DIM_NOT(3);
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);    
    cmf::CreateParallelContext(&argc, &argv);
    
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    double coords[2];
    coords[0] = 0.2 + 0.4*cmf::globalGroup.Rank();
    coords[1] = 0.5;
    double radius = 0.1;
    cmf::AxisAlignedLongCylinder cyl(coords, radius, 2);
    domain.Blocks()->SetRefineLimitCriterion([](cmf::RefinementTreeNode* n){return (n->GetLevel() > 1);});
    for (cmf::BlockIterator lb(domain.Blocks(), cmf::BlockFilters::Terminal); lb.HasNext(); lb++)
    {
        if (cyl.BoxIntersectsBoundary(lb.Node()->GetBlockBounds())) lb.Node()->Refine(7);
    }
    /*
    bool syncedHash = cmf::globalGroup.HasSameValue(domain.Blocks()->GetHash());
    std::cout << syncedHash << ", " << domain.Blocks()->GetHash() << ", " << cmf::globalGroup.Rank() << ", " << inputInfo.refinementConstraintType << std::endl;
    domain.Blocks()->OutputDebugVtk("output/mesh" + std::to_string(cmf::globalGroup.Rank()) + ".vtk");*/
    return 0;
}