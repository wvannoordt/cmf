#include <iostream>
#include <cmath>
#include "Anaptric.h"
bool limit(Anaptric::RefinementTreeNode* block)
{
    return (block->GetLevel() > 2);
}

int main(int argc, char** argv)
{
    Anaptric::Initialize();
    Anaptric::ReadInput("input.ptl");
    Anaptric::RefinementBlock domain("Domain");
    Anaptric::RefinementBlock domain2("Domain2");
    std::string filename = "output/main.tex";
    Anaptric::TikzObject picture;
    picture.Open(filename);
	picture.SetClip(0,0,1,1);
    domain.Render(&picture);
    domain2.Render(&picture);
    picture.Close();
    Anaptric::Finalize();
    return 0;
}
