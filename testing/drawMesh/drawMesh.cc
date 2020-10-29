#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"

int main(int argc, char** argv)
{
    EXIT_WARN_IF_DIM_NOT(2);
    cmf::Initialize();
    cmf::ReadInput("input.ptl");
    cmf::globalDebugLevel = 4;
    cmf::globalTrackOutputOrigins = true;
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    double bounds[4];
    double localBounds[4];
    for (int i = 0; i < 4; i++) bounds[i] = inputInfo.blockBounds[i];

    double point[2];
    point[0] = 0.5*(bounds[0] + bounds[1]) + 0.3;
    point[1] = 0.5*(bounds[2] + bounds[3]);

    cmf::AxisAlignedLongCylinder cylinder(point, 0.3, 2);
    domain.Blocks()->SetRefineLimitCriterion([](cmf::RefinementTreeNode* n){return (n->GetLevel() > 4);});
    for (cmf::BlockIterator i(domain.Blocks(), cmf::BlockFilters::Terminal); i.HasNext(); i++)
    {
        double* localBoundsBlock = i.Node()->GetBlockBounds();
        for (int i = 0; i < 4; i++) localBounds[i] = localBoundsBlock[i];
        if (cylinder.BoxIntersectsBoundary(localBounds)) i.Node()->Refine(7);
    }

    cmf::pxtype black = {0, 0, 0, 255};
    cmf::pxtype gray = {45, 45, 45, 255};
    cmf::pxtype t_red = {0, 0, 255, 145};
    cmf::pxtype white = {255, 255, 255, 255};
    cmf::pxtype green = {0, 255, 0, 255};

    cmf::PngImage img(512,2048);
    cmf::PngDrawer canvas(&img);
    canvas.SetCoordinateSystem(bounds[0], bounds[1], bounds[2], bounds[3]);
    canvas.Fill(black);
    int u = 0;
    for (cmf::BlockIterator i(domain.Blocks(), cmf::BlockFilters::Terminal); i.HasNext(); i++)
    {
        double* localBoundsBlock = i.Node()->GetBlockBounds();
        for (int i = 0; i < 4; i++)
        {
            localBounds[i] = localBoundsBlock[i];
            //Test the output system
            WriteLine(2, "Copying buffer value " + std::to_string(i))
        }
        canvas.OutlineBox(localBounds[0], localBounds[1], localBounds[2], localBounds[3], 0, green, 1);
        WriteLine(1, "Draw block " + std::to_string(u++));
    }
    img.Write("output/domain.png");
    cmf::Finalize();
}
