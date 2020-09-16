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
        coords[0] = 0.1;
        coords[1] = 0.1;
        coords[2] = 0.00001;
        //domain.RefineAt(coords, 7);
        //domain.RefineAt(coords, 7);
        domain.RefineRandom();
        domain.RefineRandom();
        std::string filename = "output/domain.vtk";
        domain.OutputDebugVtk(filename);
        Anaptric::Finalize();
    )
    return 0;
}
