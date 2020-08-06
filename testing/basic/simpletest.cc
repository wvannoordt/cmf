#include <iostream>
#include "gTree.h"
int main(int argc, char** argv)
{
    gTree::Initialize();    
    gTree::ReadInput("input.ptl");
    gTree::RefinementBlock domain("Domain");
    double coords[2];
    coords[0] = 2.999;
    coords[1] = 2.999;
    //domain.RefineRandom();
    domain.RefineAt(coords, 3);
    domain.RefineAt(coords, 3);
    //domain.RefineAt(coords, 3);
    //domain.RefineAt(coords, 3);
    domain.Render("output/main.tex");
    gTree::Finalize();
    return 0;
}
