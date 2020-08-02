#include <iostream>
#include "gTree.h"
int main(int argc, char** argv)
{
    gTree::Initialize();    
    gTree::ReadInput("input.ptl");
    gTree::RefinementBlock domain("Domain");
    double coords[2];
    coords[0] = 2.345789;
    coords[1] = 3.14156;
    //domain.RefineRandom();
    //domain.RefineRandom();
    //domain.RefineRandom();
    //domain.RefineRandom();
    domain.RefineAt(coords, 1);
    domain.RefineAt(coords, 2);
    domain.RefineAt(coords, 3);
    domain.RefineAt(coords, 3);
    domain.Render("output/main.tex");
    gTree::Finalize();
    return 0;
}
