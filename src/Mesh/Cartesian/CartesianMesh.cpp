#include "CartesianMesh.h"
#include "CmfScreen.h"
#include "DebugTools.hx"
#include "AmrFcnTypes.h"
#include "Utils.hx"
#include "BlockIndexing.h"
#include "BlockArray.h"
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
    
    CartesianMeshArray& CartesianMesh::DefineVariable(std::string name, size_t elementSize)
    {
        ArrayInfo info;
        info.name = name;
        info.rank = 0;
        info.elementSize = elementSize;
        return DefineVariable(info, BlockFilters::Terminal);
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
        CartesianMeshArray& coordArray = *(arrayHandler->CreateNewVariable(info, filter));
        for (BlockIterator lb(this, filter, IterableMode::parallel); lb.HasNext(); lb++)
        {
            BlockInfo info = this->GetBlockInfo(lb);
            BlockArray<double> coords = coordArray[lb];
            for (cell_t k = info.kmin; k < info.kmax; k++)
            {
                for (cell_t j = info.jmin; j < info.jmax; j++)
                {
                    for (cell_t i = info.imin; i < info.imax; i++)
                    {
                        coordval[0] = info.blockBounds[0]+((double)i + 0.5)*info.dx[0];
                        coordval[1] = info.blockBounds[2]+((double)j + 0.5)*info.dx[1];
                        coordval[2] = 0.0;
#if(CMF_IS3D)
                        coordval[2] = info.blockBounds[4]+((double)k + 0.5)*info.dx[2];
#endif
                        coords(i, j, k) = coordval[direction];
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
            output.blockBounds[2*d] = blockBounds[2*d];
            output.blockBounds[2*d+1] = blockBounds[2*d+1];
            output.blockSize[d] = blockBounds[2*d+1] - blockBounds[2*d];
            output.dx[d] = output.blockSize[d] / meshDataDim[d];
            output.dxInv[d] = 1.0 / output.dx[d];
            output.dataDim[d] = meshDataDim[d];
            output.totalDataDim[d] = meshDataDim[d] + 2*exchangeDim[d];
            output.exchangeDim[d] = exchangeDim[d];
        }
        output.exchangeI = exchangeDim[0];
        output.exchangeJ = exchangeDim[1];
        output.exchangeK = (CMF_IS3D?exchangeDim[1+CMF_IS3D]:0);
        output.imin = 0;
        output.imax = meshDataDim[0];
        output.jmin = 0;
        output.jmax = meshDataDim[1];
        output.kmin = 0;
        output.kmax = (1-CMF_IS3D) + CMF_IS3D*meshDataDim[1+CMF_IS3D];
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
