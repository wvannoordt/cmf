#include <string>
#include <iostream>
#include "PropTreeLib.h"
#include "gTree.h"
#include "RefinementBlock.h"
#include "Config.hx"

namespace gTree
{
    RefinementBlock::RefinementBlock(std::string title)
    {
        localInput.SetAsSubtree(mainInput[title]);
        localInput["blockDim"].MapTo(&blockDim) = new PropTreeLib::Variables::PTLStaticIntegerArray(DIM, "Base block dimensions");
        localInput["blockBounds"].MapTo(&blockBounds) = new PropTreeLib::Variables::PTLStaticDoubleArray(2*DIM, "Base block bounds");
        localInput.StrictParse();
        blockElemSize = 1;
        for (int i = 0; i < DIM; i++) blockElemSize*=blockDim[i];
    }

    RefinementBlock::~RefinementBlock(void)
    {

    }

    void RefinementBlock::Print(void)
    {
        localInput.DebugPrint();
    }
}
