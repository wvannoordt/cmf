#include <iostream>
#include "gTree.h"
int main(int argc, char** argv)
{
    gTree::Initialize();
    gTree::ReadInput("input.ptl");
    gTree::RefinementBlock domain("Domain");
    domain.Render("output/main.tex");
    gTree::Finalize();
    return 0;
}
