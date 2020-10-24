#include <iostream>
#include <cmath>
#include "cmf.h"
using cmf::cmfout;
using cmf::cmfendl;

int main(int argc, char** argv)
{
    if (3 != cmf::GetDim())
    {
        cmf::cmfout << "WARNING: skipping test case in file " << __FILE__ << ": dimensions incompatible." << cmf::cmfendl;
        return 0;
    }
    cmf::Initialize();
    PropTreeLib::PropertyTree userInput;
    bool allNeighs;
    bool level0;
    bool doRefinement;
    int testDir;
    int testVal;
    //cmfout.AddFileToStream("output/cmf.log");
    std::string outputName = "output/neigh.vtk";
    
    userInput.SetAsSubtree(cmf::mainInput["user"]);
    userInput["allNeighs"].MapTo(&allNeighs) = new PropTreeLib::Variables::PTLBoolean(false, "dummy");
    userInput["level0"].MapTo(&level0) = new PropTreeLib::Variables::PTLBoolean(false, "dummy");
    userInput["doRefinement"].MapTo(&doRefinement) = new PropTreeLib::Variables::PTLBoolean(false, "dummy");
    userInput["testDir"].MapTo(&testDir) = new PropTreeLib::Variables::PTLEnum("x", "x:y:z", "dummy");
    userInput["testVal"].MapTo(&testVal) = new PropTreeLib::Variables::PTLInteger(-1, "dummy");
    //userInput["outputName"].MapTo(&outputName) = new PropTreeLib::Variables::PTLString("output/neigh.vtk", "dummy");
    cmf::ReadInput("input.ptl");
    userInput.StrictParse();
    cmf::mainInput.DebugPrint();
    cmf::CartesianMeshInputInfo inputInfo("Domain", cmf::mainInput);
    cmf::CartesianMesh domain(inputInfo);
    double coords[3];
    coords[0] = 0.001;
    coords[1] = 0.001;
    coords[2] = 0.001;
    if (doRefinement) domain.Blocks()->RefineAt(coords, 7);
    coords[0] = 0.999;
    coords[1] = 0.999;
    coords[2] = 0.999;
    cmf::RefinementTreeNode* targetnode = domain.Blocks()->GetNodeAt(coords);
    int numBlocks = 1;
    for (cmf::NeighborIterator i(targetnode); i.Active(); i++)
    {
        if ((i.Edge().edgeVector[testDir] == testVal) || allNeighs)
        {
            if (!level0 || (i.Node()->GetLevel()==0))
            {
                numBlocks++;
            }
        }
    }
    cmf::VtkFile output(outputName, cmf::VtkFormatType::ascii, cmf::VtkTopologyType::unstructuredGrid);
    output.Mesh()->Component("DATASET")->SetAttribute("numPoints", 8*numBlocks);
    output.Mesh()->Component("DATASET")->SetAttribute("bufferCount", 3*8*numBlocks);
    output.Mesh()->Component("DATASET")->SetAttribute("stride", 3);
    output.Mesh()->Component("CELLS")->SetAttribute("numPoints", numBlocks);
    output.Mesh()->Component("CELLS")->SetAttribute("bufferCount", 9*numBlocks);
    output.Mesh()->Component("CELLS")->SetAttribute("totalEntries", 9*numBlocks);
    output.Mesh()->Component("CELLS")->SetAttribute("stride", 9);
    output.Mesh()->Component("CELL_TYPES")->SetAttribute("numPoints", numBlocks);
    output.Mesh()->Component("CELL_TYPES")->SetAttribute("bufferCount", numBlocks);
    output.Mesh()->Component("CELL_TYPES")->SetAttribute("stride", 1);
    cmf::VtkBuffer points(output.Mesh()->Component("DATASET"));
    cmf::VtkBuffer edges(output.Mesh()->Component("CELLS"));
    cmf::VtkBuffer cellTypes(output.Mesh()->Component("CELL_TYPES"));
    int count = 0;
    targetnode->WriteBlockDataToVtkBuffers(points, edges, cellTypes, &count);
    int neighcount = 0;
    for (cmf::NeighborIterator i(targetnode); i.Active(); i++)
    {
        neighcount++;
        if ((i.Edge().edgeVector[testDir] == testVal) || allNeighs)
        {
            if (!level0 || (i.Node()->GetLevel()==0))
            {
                i.Node()->WriteBlockDataToVtkBuffers(points, edges, cellTypes, &count);
            }
        }
    }
    //cmfout << "Neighcount " << neighcount << cmfendl;
    //cmfout << "Realneighcound " << targetnode->NumberOfNeighbors() << cmfendl;
    output.Write();
    cmf::Finalize();
    return 0;
}
