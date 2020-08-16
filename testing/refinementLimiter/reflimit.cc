#include <iostream>
#include <cmath>
#include "gTree.h"
int main(int argc, char** argv)
{
    gTree::Initialize();    
    gTree::ReadInput("input.ptl");
    gTree::RefinementBlock domain("Domain");   
    
    std::string filename = "output/main.tex";
    gTree::TikzObject picture;
    picture.Open(filename);
    domain.Render(&picture);
    picture.Close();
    gTree::Finalize();
    return 0;
}
