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
        double coords[3];
        coords[0] = 0.0001;
        coords[1] = 0.0001;
        coords[2] = 0.0001;
        domain.RefineRandom();
        domain.RefineRandom();
        domain.RefineRandom();
        domain.RefineRandom();
        std::string filename = "output/domain.vtk";
        domain.OutputDebugVtk(filename);
        Anaptric::Finalize();
    )
    return 0;
}
