#include <string>
#include <iostream>
#include "PropTreeLib.h"
#include "gTree.h"
#include "RefinementBlock.h"
#include "RefinementTreeNode.h"
#include "Config.hx"
#include "TikzObject.h"

namespace gTree
{
    RefinementBlock::RefinementBlock(std::string title)
    {
        localInput.SetAsSubtree(mainInput[title]);
        localInput["blockDim"].MapTo(&blockDim) = new PropTreeLib::Variables::PTLStaticIntegerArray(DIM, "Base block dimensions");
        localInput["blockBounds"].MapTo(&blockBounds) = new PropTreeLib::Variables::PTLStaticDoubleArray(2*DIM, "Base block bounds");
        localInput.StrictParse();
        totalNumTrunks = 1;
        for (int i = 0; i < DIM; i++) totalNumTrunks*=blockDim[i];
        Allocate();
    }

    RefinementBlock::~RefinementBlock(void)
    {
        this->Destroy();
    }

    void RefinementBlock::Allocate(void)
    {
        deallocTrunks = true;
        trunks = new RefinementTreeNode* [totalNumTrunks];
        for (int i = 0; i < totalNumTrunks; i++) trunks[i] = new RefinementTreeNode();
    }

    void RefinementBlock::Destroy(void)
    {
        if (deallocTrunks)
        {
            for (int i = 0; i < totalNumTrunks; i++)
            {
                trunks[i]->Destroy();
                delete trunks[i];
            }
            delete[] trunks;
        }
    }

    void RefinementBlock::Print(void)
    {
        localInput.DebugPrint();
    }

    void RefinementBlock::Render(std::string filename)
    {
        TikzObject picture;
        picture.Open(filename);
        picture.DrawGrid(blockBounds[0], blockBounds[2], blockBounds[1], blockBounds[3]);
        picture.Close();
    }
}
