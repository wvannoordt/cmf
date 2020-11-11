#include "CartesianMesh.h"
#include "CmfScreen.h"
#include "DebugTools.hx"
#include "AmrFcnTypes.h"
namespace cmf
{
    CartesianMesh::CartesianMesh(CartesianMeshInputInfo input) : ICmfMesh(input, MeshType::Cartesian)
    {
        title = input.title;
        blockDim = input.blockDim;
        blockBounds = input.blockBounds;
        refinementConstraintType = input.refinementConstraintType;
        blocks = new RefinementBlock(blockDim, blockBounds, refinementConstraintType);
        meshDataDim = input.meshDataDim;
        exchangeDim = input.exchangeDim;
        arrayHandler = new CartesianMeshArrayHandler(this);
    }

    RefinementBlock* CartesianMesh::Blocks(void)
    {
        return blocks;
    }

    ICmfMeshArrayHandler* CartesianMesh::GetArrayHandler(void)
    {
        return arrayHandler;
    }

    CartesianMeshArray& CartesianMesh::DefineVariable(std::string name)
    {
        ArrayInfo info;
        info.name = name;
        info.rank = 0;
        info.elementSize = sizeof(double);
        return DefineVariable(info, BlockFilters::Terminal);
    }
    
    CartesianMeshArray& CartesianMesh::DefineVariable(std::string name, NodeFilter_t filter)
    {
        ArrayInfo info;
        info.name = name;
        info.rank = 0;
        info.elementSize = sizeof(double);
        return DefineVariable(info, filter);
    }
    
    CartesianMeshArray& CartesianMesh::DefineVariable(ArrayInfo info)
    {
        return DefineVariable(info, BlockFilters::Terminal);
    }
    
    CartesianMeshArray& CartesianMesh::DefineVariable(ArrayInfo info, NodeFilter_t filter)
    {
        return *(arrayHandler->CreateNewVariable(info, filter));
    }
    
    BlockInfo CartesianMesh::GetBlockInfo(RefinementTreeNode* node)
    {
        BlockInfo output;
        double* blockBounds = node->GetBlockBounds();
        for (int d = 0; d < CMF_DIM; d++)
        {
            output.dataDim[d] = meshDataDim[d];
            output.exchangeDim[d] = exchangeDim[d];
            output.blockBounds[2*d] = blockBounds[2*d];
            output.blockBounds[2*d+1] = blockBounds[2*d+1];
            output.blockSize[d] = blockBounds[2*d+1] - blockBounds[2*d];
            output.dx[d] = output.blockSize[d] / meshDataDim[d];
            output.dxInv[d] = 1.0 / output.dx[d];
            output.totalDataDim[d] = meshDataDim[d]+2*exchangeDim[d];
        }
        return output;
    }
    
    BlockInfo CartesianMesh::GetBlockInfo(BlockIterator& blockIter)
    {
        return GetBlockInfo(blockIter.Node());
    }
    
    size_t CartesianMesh::Size(void)
    {
        return blocks->Size();
    }
    
    std::vector<RefinementTreeNode*>* CartesianMesh::GetAllNodes(void)
    {
        return blocks->GetAllNodes();
    }

    CartesianMesh::~CartesianMesh(void)
    {
        delete arrayHandler;
        delete blocks;
    }
}
