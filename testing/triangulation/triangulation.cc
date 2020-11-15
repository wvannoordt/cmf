#include <iostream>
#include <cmath>
#include "cmf.h"

std::string istr(int i)
{
	std::string output = std::to_string(i);
	while (output.length() < 5)
	{
		output = "0" + output;
	}
	return output;
}

cmf::SurfaceTriangulation surface;
bool isect(cmf::RefinementTreeNode* n)
{
    double* localBoundsBlock = n->GetBlockBounds();
    if (!(n->IsTerminal())) return false;
    return (surface.BoxIntersectsBoundary(localBoundsBlock));
}

int main(int argc, char** argv)
{
	std::cout << "\n\nWARNING: Skipping triangulation test as no meshes are to be committed";
	std::cout << " to the repository.\n" << std::endl;
	return 0;
    cmf::Initialize();
    cmf::ReadInput("input.ptl");
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
	std::string filenameSTL = "goat.stl";
	std::cout << "READING " << filenameSTL << std::endl;
    surface.DefineTriangulation(filenameSTL);
    domain.Blocks()->SetRefineLimitCriterion([](cmf::RefinementTreeNode* n){return (n->GetLevel() > 6);});
    for (cmf::BlockIterator i(domain.Blocks(), cmf::BlockFilters::Terminal); i.HasNext(); i++)
    {
        double* localBoundsBlock = i.Node()->GetBlockBounds();
        if (surface.BoxIntersectsBoundary(localBoundsBlock)) i.Node()->Refine(7);
    }
	
	std::cout << "OUTPUT VTK" << std::endl;
    domain.Blocks()->OutputDebugVtk("output/domain_007.vtk", isect);
    cmf::Finalize();
    return 0;
}
