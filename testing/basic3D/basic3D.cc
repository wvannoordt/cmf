#include <iostream>
#include <cmath>
#include "Anaptric.h"
int main(int argc, char** argv)
{
    __only3d
    (
        Anaptric::Initialize();
        Anaptric::ReadInput("input.ptl");
        Anaptric::RefinementBlock domain("Domain");
        std::string filename = "output/domain.vtk";
        Anaptric::Finalize();
    )
    return 0;
}
