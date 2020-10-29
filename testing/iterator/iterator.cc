#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
using cmf::cmfout;
using cmf::cmfendl;
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
    int p = 0;
    for (cmf::BlockIterator i(domain.Blocks()); i.HasNext(); ++i)
    {
        cmfout << "Block " << i << cmfendl;
        if (p==18) i.Node()->Refine(7);
        p++;
    }
    cmf::Finalize();
    return 0;
}
