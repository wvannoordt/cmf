#include <iostream>
#include <cmath>
#include "cmf.h"
bool limit(cmf::RefinementTreeNode* block)
{
    return (block->GetLevel() > 2);
}

int main(int argc, char** argv)
{
    if (CMF_DIM != cmf::GetDim())
    {
        cmf::cmfout << "WARNING: skipping test case in file " << __FILE__ << ": dimensions incompatible." << cmf::cmfendl;
        return 0;
    }
    cmf::Initialize();
    cmf::ReadInput("input.ptl");
    cmf::CartesianMeshInputInfo inputInfo("Domain", cmf::mainInput);
    cmf::CartesianMesh domain(inputInfo);
    cmf::CartesianMeshInputInfo inputInfo2("Domain2", cmf::mainInput);
    cmf::CartesianMesh domain2(inputInfo2);
    std::string filename = "output/main.tex";
    cmf::TikzObject picture;
    picture.Open(filename);
	picture.SetClip(0,0,1,1);
    domain.Blocks()->Render(&picture);
    domain2.Blocks()->Render(&picture);
    picture.Close();
    cmf::Finalize();
    return 0;
}
