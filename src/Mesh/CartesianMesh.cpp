#include "CartesianMesh.h"

namespace cmf
{
    CartesianMesh::CartesianMesh(CartesianMeshInputInfo input): ICmfMesh(input)
    {   
        title = input.title;        
        blockDim = input.blockDim;
        blockBounds = input.blockBounds;
        refinementConstraintType = input.refinementConstraintType;
        blocks = new RefinementBlock(blockDim, blockBounds, refinementConstraintType);
    }
    
    RefinementBlock* CartesianMesh::Blocks(void)
    {
        return blocks;
    }
    
    CartesianMesh::~CartesianMesh(void)
    {
        delete blocks;
    }
}