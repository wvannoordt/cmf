#include <iostream>
#include <cmath>
#include "cmf.h"
int main(int argc, char** argv)
{
    if (2 != cmf::GetDim())
    {
        cmf::cmfout << "WARNING: skipping test case in file " << __FILE__ << ": dimensions incompatible." << cmf::cmfendl;
        return 0;
    }
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
