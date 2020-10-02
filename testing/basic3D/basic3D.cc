#include <iostream>
#include <cmath>
#include "cmf.h"
int main(int argc, char** argv)
{
    __only3d
    (
        cmf::Initialize();
        cmf::ReadInput("input.ptl");
        cmf::RefinementBlock domain("Domain");
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
        cmf::Finalize();
    )
    return 0;
}
