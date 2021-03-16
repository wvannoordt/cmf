#ifndef CMF_BLOCK_INFO_H
#define CMF_BLOCK_INFO_H
#include "ICmfHasBlockBoundIndices.h"
namespace cmf
{
    /// @brief A struct general block information that can be passed to a numerical kernel, used for indexing and computing coorintes
    /// @author WVN
    struct BlockInfo : public ICmfHasBlockBoundIndices
    {        
        //@brief bounding box of the current block
        double blockBounds[2*CMF_DIM];
        
        //@brief The dimensions of the box in each coordinate
        double blockSize[CMF_DIM];
        
        //@brief The number of cells (excluding the exchange cells) in each direction
        double dataDim[CMF_DIM];
        
        //@brief The number of exchange cells
        double exchangeDim[CMF_DIM];
        
        //@brief The number of cells (including the exchange cells) in each direction
        double totalDataDim[CMF_DIM];
        
        //@brief the mesh spacing of the current block
        double dx[CMF_DIM];
        
        //@brief reciporocal of the mesh spacing of the current block
        double dxInv[CMF_DIM];
    };
}

#endif