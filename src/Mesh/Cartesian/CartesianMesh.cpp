#include "CartesianMesh.h"
#include "CmfScreen.h"
#include "DebugTools.hx"
#include "AmrFcnTypes.h"
#include "Utils.hx"
#include "BlockIndexing.h"
namespace cmf
{
    CartesianMesh::CartesianMesh(CartesianMeshInputInfo input) : ICmfMesh(input, MeshType::Cartesian)
    {
        title = input.title;
        blockDim = input.blockDim;
        blockBounds = input.blockBounds;
        refinementConstraintType = input.refinementConstraintType;
        blocks = new RefinementBlock(blockDim, blockBounds, refinementConstraintType, input.periodicRefinement);
        meshDataDim = input.meshDataDim;
        exchangeDim = input.exchangeInfo.exchangeDim;
        arrayHandler = new CartesianMeshArrayHandler(this);
        meshGroup = &globalGroup;
        hasParallelPartition = false;
        partition = NULL;
        CreateParallelPartition(input.partitionInfo);
        arrayHandler->CreateExchangeHandler(input.exchangeInfo);
    }

    RefinementBlock* CartesianMesh::Blocks(void)
    {
        return blocks;
    }

    ICmfMeshArrayHandler* CartesianMesh::GetArrayHandler(void)
    {
        return arrayHandler;
    }
    
    CartesianMeshParallelPartition* CartesianMesh::GetPartition(void)
    {
        return partition;
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
    
    CartesianMeshArray& CartesianMesh::DefineVariable(std::string name, size_t elementSize, NodeFilter_t filter)
    {
        ArrayInfo info;
        info.name = name;
        info.rank = 0;
        info.elementSize = elementSize;
        return DefineVariable(info, filter);
    }
    
    CartesianMeshArray& CartesianMesh::DefineVariable(std::string name, size_t elementSize, std::initializer_list<int> arrayDimensions, NodeFilter_t filter)
    {
        ArrayInfo info;
        info.name = name;
        info.rank = arrayDimensions.size();
        if (info.rank>MAX_RANK) CmfError("A mesh array \"" + name + "\" was initialized with rank " + std::to_string(info.rank) + ", exceeding maximum rank " + std::to_string(MAX_RANK));
        int r = 0;
        for (auto dim:arrayDimensions)
        {
            info.dimensions[r++] = dim;
        }
        info.elementSize = elementSize;
        return DefineVariable(info, filter);
    }
    
    CartesianMeshArray& CartesianMesh::DefineVariable(std::string name, size_t elementSize, std::initializer_list<int> arrayDimensions)
    {
        ArrayInfo info;
        info.name = name;
        info.rank = arrayDimensions.size();
        if (info.rank>MAX_RANK) CmfError("A mesh array \"" + name + "\" was initialized with rank " + std::to_string(info.rank) + ", exceeding maximum rank " + std::to_string(MAX_RANK));
        int r = 0;
        for (auto dim:arrayDimensions)
        {
            info.dimensions[r++] = dim;
        }
        info.elementSize = elementSize;
        return DefineVariable(info, BlockFilters::Terminal);
    }
    
    CartesianMeshArray& CartesianMesh::DefineVariable(ArrayInfo info)
    {
        return DefineVariable(info, BlockFilters::Terminal);
    }
    
    CartesianMeshArray& CartesianMesh::DefineVariable(ArrayInfo info, NodeFilter_t filter)
    {
        return *(arrayHandler->CreateNewVariable(info, filter));
    }
    
    CartesianMeshArray& CartesianMesh::CreateCoordinateVariable(NodeFilter_t filter, int direction)
    {
        ArrayInfo info;
        std::string name = "";
        switch (direction)
        {
            case 0: {name = "x"; break;}
            case 1: {name = "y"; break;}
            case 2: {name = "z"; break;}
            default: {CmfError("Attempted to create coordinate variable with invalid direction ID " + std::to_string(direction));}
        }
        info.name = name;
        info.rank = 0;
        info.elementSize = sizeof(double);
        double coordval[3];
        int idx[CMF_DIM];
        CartesianMeshArray& coordArray = *(arrayHandler->CreateNewVariable(info, filter));
        for (BlockIterator lb(this, filter, IterableMode::parallel); lb.HasNext(); lb++)
        {
            BlockInfo info = this->GetBlockInfo(lb);
            double* coordBuffer = (double*)coordArray[lb];
            cmf_pkloop(idx[2], info.exchangeDim[2], info)
            {
                cmf_pjloop(idx[1], info.exchangeDim[1], info)
                {
                    cmf_piloop(idx[0], info.exchangeDim[0], info)
                    {
                        __dloop(coordval[d] = info.blockBounds[2*d]+((double)idx[d] + 0.5)*info.dx[d]);
#if(!CMF_IS3D)
                        coordval[2] = 0.0;
#endif
                        coordBuffer[cmf_idx(idx[0], idx[1], idx[2], info)] = coordval[direction];
                    }
                }
            }
        }
        return coordArray;
    }
    
    CartesianMeshArray& CartesianMesh::CreateCoordinateVariable(int direction)
    {
        return CreateCoordinateVariable(BlockFilters::Terminal, direction);
    }
    
    RefinementBlock* CartesianMesh::GetRefinementBlockObject(void)
    {
        return Blocks();
    }
    
    bool CartesianMesh::ParallelPartitionContainsNode(RefinementTreeNode* node)
    {
        if (!partition) return true;
        return partition->Mine(node);
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
    
    CartesianMeshParallelPartition* CartesianMesh::CreateParallelPartition(CartesianMeshParallelPartitionInfo& partitionInfo)
    {
        hasParallelPartition = true;
        partition = new CartesianMeshParallelPartition(this, partitionInfo);
        return partition;
    }
    
    CartesianMeshArray& CartesianMesh::operator [] (std::string name)
    {
        return *(arrayHandler->GetVariable(name));
    }
    
    void CartesianMesh::AssertSynchronizeBlocks(void)
    {
        bool syncedHash = meshGroup->HasSameValue(blocks->GetHash());
        if (!syncedHash)
        {
            CmfError("AssertSynchronizeBlocks failed in CartesianMesh with name \"" + title + "\": Ensure that refinements happen on every rank!!");
        }
        WriteLine(4, "Cartesian mesh \"" + title + "\" is sychronized");
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
    
    std::string CartesianMesh::GetTitle(void)
    {
        return title;
    }

    CartesianMesh::~CartesianMesh(void)
    {
        delete arrayHandler;
        delete blocks;
        if (hasParallelPartition)
        {
            hasParallelPartition = false;
            delete partition;
        }
    }
}
