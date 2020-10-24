#include <iostream>
#include <cmath>
#include "cmf.h"
int main(int argc, char** argv)
{
    __only3d
    (
        cmf::Initialize();
        cmf::ReadInput("input.ptl");
        cmf::CartesianMeshInputInfo inputInfo("Domain", cmf::mainInput);
        cmf::CartesianMesh domain(inputInfo);
        double coords[3];
        coords[0] = 0.1;
        coords[1] = 0.1;
        coords[2] = 0.00001;
        //domain.Blocks()->RefineAt(coords, 7);
        //domain.Blocks()->RefineAt(coords, 7);
        domain.Blocks()->RefineRandom();
        domain.Blocks()->RefineRandom();
        std::string filename = "output/domain.vtk";
        domain.Blocks()->OutputDebugVtk(filename);
        cmf::Finalize();
    )
    return 0;
}
