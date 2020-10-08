#include <iostream>
#include <cmath>
#include "cmf.h"
using cmf::cmfout;
using cmf::cmfendl;

int main(int argc, char** argv)
{
    __only3d
    (
        cmf::Initialize();
        PropTreeLib::PropertyTree user;
        bool allNeighs;
        bool doRefinement;
        int testDir;
        int testVal;
        user.SetAsSubtree(cmf::mainInput["user"]);
        user["allNeighs"].MapTo(&allNeighs) = new PropTreeLib::Variables::PTLBoolean(false, "dummy");
        user["doRefinement"].MapTo(&doRefinement) = new PropTreeLib::Variables::PTLBoolean(false, "dummy");
        user["testDir"].MapTo(&testDir) = new PropTreeLib::Variables::PTLEnum("x", "x:y:z", "dummy");
        user["testVal"].MapTo(&testVal) = new PropTreeLib::Variables::PTLInteger(-1, "dummy");
        cmf::ReadInput("input.ptl");
        user.StrictParse();
        cmf::RefinementBlock domain("Domain");
        double coords[3];
        coords[0] = 0.5001;
        coords[1] = 0.5001;
        coords[2] = 0.5001;
        if (doRefinement) domain.RefineAt(coords, 7);
        coords[0] = 0.501;
        coords[1] = 0.501;
        coords[2] = 0.501;
        CmfError("test");
        cmf::RefinementTreeNode* targetnode = domain.GetNodeAt(coords);
        int numBlocks = 1;
        for (cmf::NeighborIterator i(targetnode); i.Active(); i++)
        {
            if ((i.Edge().edgeVector[testDir] == testVal) || allNeighs)
            {
                numBlocks++;
            }
        }
        std::string filename = "output/neigh.vtk";
        cmf::VtkFile output(filename, cmf::VtkFormatType::ascii, cmf::VtkTopologyType::unstructuredGrid);
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
        for (cmf::NeighborIterator i(targetnode); i.Active(); i++)
        {
            if ((i.Edge().edgeVector[testDir] == testVal) || allNeighs)
            {
                i.Node()->WriteBlockDataToVtkBuffers(points, edges, cellTypes, &count);
            }
        }
        output.Write();
        cmf::Finalize();
    )
    return 0;
}
