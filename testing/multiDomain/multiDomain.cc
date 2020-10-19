#include <iostream>
#include <cmath>
#include "cmf.h"
bool limit(cmf::RefinementTreeNode* block)
{
    return (block->GetLevel() > 2);
}

int main(int argc, char** argv)
{
    __only2d
    (
        cmf::Initialize();
        cmf::ReadInput("input.ptl");
        cmf::CartesianMesh domain("Domain");
        cmf::CartesianMesh domain2("Domain2");
        std::string filename = "output/main.tex";
        cmf::TikzObject picture;
        picture.Open(filename);
    	picture.SetClip(0,0,1,1);
        domain.Blocks()->Render(&picture);
        domain2.Blocks()->Render(&picture);
        picture.Close();
        cmf::Finalize();
    )
    return 0;
}
