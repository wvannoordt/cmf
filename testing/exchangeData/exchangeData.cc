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
    domain.Blocks()->RefineRandom();
    domain.Blocks()->RefineRandom();
    cmf::CartesianMeshArray dist = domain.DefineVariable("distance");
    cmf::PngImage img(2048,2048);
    cmf::PngDrawer canvas(&img);
    double* boundsL = inputInfo.blockBounds;
    canvas.SetCoordinateSystem(boundsL[0], boundsL[1], boundsL[2], boundsL[3]);
    cmf::pxtype black = {0, 0, 0, 255};
    cmf::pxtype lightBlue = {255, 150, 100, 255};
    canvas.Fill(black);
    for (cmf::BlockIterator lb(&dist, cmf::BlockFilters::Terminal); lb.HasNext(); lb++)
    {
        double* bounds = lb.Node()->GetBlockBounds();
        canvas.OutlineBox(bounds[0], bounds[1], bounds[2], bounds[3], 0, lightBlue, 3);
    }
    img.Write("output/domain.png");
    cmf::Finalize();
    return 0;
}
