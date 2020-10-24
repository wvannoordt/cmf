#ifndef CMF_H
#define CMF_H

#include "CmfError.h"
#include <string>
#include "Config.h"
#include "PropTreeLib.h"
#include "CartesianMesh.h"
#include "RefinementBlock.h"
#include "RefinementTreeNode.h"
#include "NeighborIterator.h"
#include "VtkFile.h"
#include "PngImage.h"
#include "PngDrawer.h"
namespace cmf
{
    extern PropTreeLib::PropertyTree mainInput;
    
    /// @brief Initializes CMF. Currently does nothing. It is intended that this function does nothing.
    /// @author WVN
    void Initialize(void);
    
    /// @brief Reads the input file (PTL format) specified by filename. PTL format is specific to <a href="https://github.com/wvannoordt/PropTreeLib">PropTreeLib</a>.
    /// @param filename The name if the input file to be read
    /// @author WVN
    void ReadInput(std::string filename);
    
    /// @brief Finalizes CMF. Currently does nothing. It is intended that this function does nothing.
    /// @author WVN
    void Finalize(void);
    
    /// @brief Returns CMF_DIM, used to check that the host code is using the correct dimension (determined at compile time) of CMF.\see Config.h
    /// @author WVN
    int GetDim(void);
}

#endif
