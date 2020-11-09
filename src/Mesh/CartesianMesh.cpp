#include "CartesianMesh.h"
#include "CmfScreen.h"
#include "DebugTools.hx"
#include "AmrFcnTypes.h"
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

    CartesianMesh::~CartesianMesh(void)
    {
        delete arrayHandler;
        delete blocks;
    }
}
