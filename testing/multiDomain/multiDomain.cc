#include <iostream>
#include <cmath>
#include "gTree.h"
bool limit(gTree::RefinementTreeNode* block)
{
    return (block->GetLevel() > 2);
}

int main(int argc, char** argv)
{
    gTree::Initialize();
    gTree::ReadInput("input.ptl");
    gTree::RefinementBlock domain("Domain");
    gTree::RefinementBlock domain2("Domain2");
    std::string filename = "output/main.tex";
    gTree::TikzObject picture;
    picture.Open(filename);
	picture.SetClip(0,0,1,1);
    domain.Render(&picture);
    domain2.Render(&picture);
    picture.Close();
    gTree::Finalize();
    return 0;
}
