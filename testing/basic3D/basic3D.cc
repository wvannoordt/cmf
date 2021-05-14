#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
int main(int argc, char** argv)
{
    EXIT_WARN_IF_DIM_NOT(3);
    cmf::Initialize();
    cmf::ReadInput("input.ptl");
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    double coords[3];
    coords[0] = 0.1;
    coords[1] = 0.1;
    coords[2] = 0.00001;
    //domain.Blocks()->RefineAt(coords, 7);
    domain.Blocks()->RefineAt(coords, 7);
    std::string filename = "output/domain.vtk";
    domain.Blocks()->OutputDebugVtk(filename);
    cmf::Finalize();
    return 0;
}
