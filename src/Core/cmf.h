#ifndef CMF_H
#define CMF_H

#include "CmfError.h"
#include <string>
#include "Config.h"
#include "PropTreeLib.h"
#include "RefinementBlock.h"
#include "RefinementTreeNode.h"
#include "NeighborIterator.h"
namespace cmf
{
    extern PropTreeLib::PropertyTree mainInput;
    void Initialize(void);
    void ReadInput(std::string filename);
    void Finalize(void);
    int GetDim(void);
}

#endif
