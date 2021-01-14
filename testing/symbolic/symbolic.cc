#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>

int main(int argc, char** argv)
{
    EXIT_WARN_IF_DIM_NOT(3);
    bool doRefinement = false;
    
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::Initialize();
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    cmf::CartesianMeshArray& x = domain.CreateCoordinateVariable(0);
    cmf::CartesianMeshArray& y = domain.CreateCoordinateVariable(1);
    cmf::CartesianMeshArray& z = domain.CreateCoordinateVariable(2);
    cmf::CartesianMeshArray& newVariable = domain.DefineVariable("newVariable");
    auto newVariable2 = x + y; // <-- goals
    std::cout << newVariable2.GetExpressionString() << std::endl;
    newVariable = newVariable2;
    cmf::SerialCartesianVtk dataVtk(domain, "output/data.vtk");
    dataVtk << x;
    dataVtk << y;
    dataVtk << z;
    // dataVtk << newVariable;
    dataVtk.Write();
    return 0;
}