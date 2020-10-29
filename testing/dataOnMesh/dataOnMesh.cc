#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
int main(int argc, char** argv)
{
    EXIT_WARN_IF_DIM_NOT(2);
    cmf::Initialize();
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    double coords[2];
    coords[0] = 0.105;
    coords[1] = 0.54;
    double radius = 0.1;
    cmf::AxisAlignedLongCylinder cyl(coords, radius, 2);
    domain.Blocks()->SetRefineLimitCriterion([](cmf::RefinementTreeNode* n){return (n->GetLevel() > 4);});

    for (cmf::BlockIterator i(domain.Blocks(), cmf::BlockFilters::Terminal); i.HasNext(); i++)
    {
        if (cyl.BoxIntersectsBoundary(i.Node()->GetBlockBounds())) i.Node()->Refine(7);
    }

    domain.DefineVariable("distance");


    int p = 0;
    std::string filename = "output/main.tex";
    cmf::TikzObject picture;
    picture.Open(filename);
	picture.SetClip(0,0,1,1);
    domain.Blocks()->Render(&picture);
    picture.PushLineType("solid", 0.03, "red");
    picture.DrawCircle(coords[0], coords[1], radius);
    picture.Close();
    cmf::Finalize();
    return 0;
}
