#include "CartesianMesh.h"
#include "CmfScreen.h"
#include "DebugTools.hx"
namespace cmf
{
    CartesianMesh::CartesianMesh(CartesianMeshInputInfo input) : ICmfMesh(input)
    {   
        title = input.title;        
        blockDim = input.blockDim;
        blockBounds = input.blockBounds;
        refinementConstraintType = input.refinementConstraintType;
        blocks = new RefinementBlock(blockDim, blockBounds, refinementConstraintType);
        meshDataDim = input.meshDataDim;
    }
    
    RefinementBlock* CartesianMesh::Blocks(void)
    {
        return blocks;
    }
    
    CartesianMeshArrayHandler* CartesianMesh::GetArrayHandler(void)
    {
        return &arrayHandler;
    }
    
    void CartesianMesh::DefineVariable(std::string name)
    {
        ArrayInfo info;
        info.name = name;
        info.rank = 0;
        arrayHandler.CreateNewVariable(info, this);
    }
    
    CartesianMesh::~CartesianMesh(void)
    {
        delete blocks;
    }
}