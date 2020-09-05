#ifndef ANAPTRIC_H
#define ANAPTRIC_H

#include <string>
#include "Config.h"
#include "PropTreeLib.h"
#include "RefinementBlock.h"
#include "RefinementTreeNode.h"
namespace Anaptric
{
    extern PropTreeLib::PropertyTree mainInput;
    void Initialize(void);
    void ReadInput(std::string filename);
    void Finalize(void);
}

#endif
