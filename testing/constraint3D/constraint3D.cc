#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#define RX 0.50001
#define RY 0.001
#define RZ 0.50001

int main(int argc, char** argv)
{
    EXIT_WARN_IF_DIM_NOT(3);
    cmf::Initialize();
    int numRef = 3;
    cmf::ReadInput("input.ptl");
    cmf::CartesianMeshInputInfo inputInfoA(cmf::mainInput["DomainA"]);
    cmf::CartesianMeshInputInfo inputInfoB(cmf::mainInput["DomainB"]);
    cmf::CartesianMesh domainA(inputInfoA);
    cmf::CartesianMesh domainB(inputInfoB);
    cmf::CartesianMesh* domains[2];
    domains[0] = &domainA;
    domains[1] = &domainB;
    char reftype = 7;
    for (int i = 0; i < 2; i++)
    {
        double coords[3];
        coords[0] = RX+i*1.1;
        coords[1] = RY;
        coords[2] = RZ;
        //domains[i]->RefineRandom();
        //domains[i]->RefineRandom();
        for (int j = 0; j < numRef; j++) domains[i]->Blocks()->RefineAt(coords, reftype);
        std::string filename = "output/domain_" + std::to_string(i) + ".vtk";
        domains[i]->Blocks()->OutputDebugVtk(filename, [](cmf::RefinementTreeNode* n){return (n->GetLevel() == 2);});
    }
    cmf::Finalize();
    return 0;
}
