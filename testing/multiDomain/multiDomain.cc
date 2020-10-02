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
        cmf::RefinementBlock domain("Domain");
        cmf::RefinementBlock domain2("Domain2");
        std::string filename = "output/main.tex";
        cmf::TikzObject picture;
        picture.Open(filename);
    	picture.SetClip(0,0,1,1);
        domain.Render(&picture);
        domain2.Render(&picture);
        picture.Close();
        cmf::Finalize();
    )
    return 0;
}
