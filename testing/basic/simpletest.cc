#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
int main(int argc, char** argv)
{
    EXIT_WARN_IF_DIM_NOT(2);
    cmf::Initialize();
    cmf::ReadInput("input.ptl");
    cmf::CartesianMeshInputInfo inputInfo("Domain", cmf::mainInput);
    cmf::CartesianMesh domain(inputInfo);
    std::string filename = "output/main.tex";
    cmf::TikzObject picture;
    picture.Open(filename);
    domain.Blocks()->Render(&picture);
    picture.Close();
    cmf::Finalize();    
    return 0;
}
