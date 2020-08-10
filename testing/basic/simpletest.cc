#include <iostream>
#include <cmath>
#include "gTree.h"
int main(int argc, char** argv)
{
    gTree::Initialize();    
    gTree::ReadInput("input.ptl");
    gTree::RefinementBlock domain("Domain");
    double coords[2];
    coords[0] = 0.5001;
    coords[1] = 0.5001;
    domain.RefineAt(coords, 3);
    domain.RefineAt(coords, 3);
    
    domain.Render("output/main.tex");
    gTree::Finalize();
    return 0;
}
