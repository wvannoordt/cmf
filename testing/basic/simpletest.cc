#include <iostream>
#include "gTree.h"
int main(int argc, char** argv)
{
    gTree::Initialize();
    gTree::ReadInput("input.ptl");
    gTree::RefinementBlock domain("Domain");
    domain.Print();
    gTree::Finalize();
    return 0;
}
