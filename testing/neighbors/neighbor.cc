#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
int main(int argc, char** argv)
{
    EXIT_WARN_IF_DIM_NOT(2);
    cmf::Initialize();
    cmf::ReadInput("input.ptl");
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    double coords[2];
    coords[0] = 0.5;
    coords[1] = 0.6;
    domain.Blocks()->RefineRandom();
    domain.Blocks()->RefineRandom();
    domain.Blocks()->RefineRandom();
    domain.Blocks()->RefineRandom();
    //domain.RefineRandom();
    //domain.RefineAt(coords, 2);
    std::string filename = "output/main.tex";
    cmf::TikzObject picture;
    picture.Open(filename);
	picture.SetClip(0,0,1,1);
    domain.Blocks()->Render(&picture);
    picture.Close();
    cmf::Finalize();
    return 0;
}
