#ifndef GTREE_H
#define GTREE_H

#include <string>
#include "PropTreeLib.h"
#include "RefinementBlock.h"
#include "RefinementTreeNode.h"
namespace gTree
{
    extern PropTreeLib::PropertyTree mainInput;
    void Initialize(void);
    void ReadInput(std::string filename);
    void Finalize(void);
}

#endif
