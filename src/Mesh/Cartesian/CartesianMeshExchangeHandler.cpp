#include "Config.h"
#include "CmfPrint.h"
#include "CartesianMeshExchangeHandler.h"
#include "CartesianMesh.h"
#include "RefinementConstraint.h"
#include "BlockIterator.h"
#include "Utils.hx"
#include "NeighborIterator.h"
#include "Vec.h"
#include "BlockArray.h"
#include "CmfPrint.h"
#include "StringUtils.h"
#include "ExchangeContextBlockData.h"
namespace cmf
{
    CartesianMeshExchangeHandler::CartesianMeshExchangeHandler(CartesianMesh* mesh_in, CartesianMeshExchangeInfo& inputInfo)
    {
        mesh = mesh_in;
        interpolationOrder = inputInfo.interpolationOrder;
        WriteLine(2, "Create exchange pattern on mesh \"" + mesh->GetTitle() + "\"");
        exchangeDim = inputInfo.exchangeDim;
        int maxExchangeDim = 0;
        __dloop(maxExchangeDim = CMFMAX(maxExchangeDim, exchangeDim[d]));
        if ((mesh->Blocks()->GetRefinementConstraintType() != RefinementConstraint::factor2CompletelyConstrained) && (maxExchangeDim>0))
        {
            CmfError("CartesianMeshExchangeHandler cannot currently be created with any RefinementConstraint other than factor2CompletelyConstrained, found \""
                + RefinementConstraintStr(mesh->Blocks()->GetRefinementConstraintType())
                + "\"!");
        }
    }
    
    DataExchangePattern* CartesianMeshExchangeHandler::CreateMeshArrayExchangePattern(CartesianMeshArray* meshArray)
    {
        if (exchanges.find(meshArray)!=exchanges.end())
        {
            WriteLine(4, strformat("Re-define exchange patters for \"{}\" on mesh \"{}\"", meshArray->variableName, mesh->title));
            DataExchangePattern* oldPattern = exchanges[meshArray];
            delete oldPattern;
            exchanges.erase(meshArray);
        }
        DataExchangePattern* newPattern = new DataExchangePattern(mesh->GetGroup());
        DefineExchangePatternsForArray(meshArray, newPattern);
        exchanges.insert({meshArray, newPattern});
        return newPattern;
    }
    
    void CartesianMeshExchangeHandler::DefineExchangePatternsForArray(CartesianMeshArray* meshArray, DataExchangePattern* pattern)
    {
        WriteLine(5, "Define exchange pattern for variable \"" + meshArray->variableName + "\" on mesh \"" + mesh->title + "\"");
        NodeFilter_t arFilter = meshArray->GetFilter();
        for (BlockIterator lb(mesh, arFilter, IterableMode::serial); lb.HasNext(); lb++)
        {
            for (NeighborIterator neigh(lb.Node()); neigh.Active(); neigh++)
            {
                if (arFilter(neigh.Node()))
                {
                    RefinementTreeNode* currentNode  = lb.Node();
                    RefinementTreeNode* neighborNode = neigh.Node();
                    
                    CreateExchangeTransaction(pattern, meshArray, currentNode, neighborNode, neigh.Edge());
                }
            }
        }
    }
    
    void CartesianMeshExchangeHandler::CreateExchangeTransaction(
        DataExchangePattern* pattern,
        CartesianMeshArray* meshArray,
        RefinementTreeNode* currentNode,
        RefinementTreeNode* neighborNode,
        NodeEdge relationship)
    {
        ExchangeContextBlockData current;
        ExchangeContextBlockData neighbor;
        
        //Retrieve partition info and block dimensions from the mesh
        current.partitionInfo  = mesh->partition->GetPartitionInfo(currentNode);
        neighbor.partitionInfo = mesh->partition->GetPartitionInfo(neighborNode);
        current.blockInfo      = mesh->GetBlockInfo(currentNode);
        neighbor.blockInfo     = mesh->GetBlockInfo(neighborNode);
        
        //Compute the size of a single array "element"
        size_t singleCellSize = SizeOfArrayType(meshArray->elementType);
        for (int i = 0; i < meshArray->rank; i++)
        {
            singleCellSize *= meshArray->dims[i];
        }
        
        //Compute array sizes for data transfer
        int niCurrent = current.blockInfo.totalDataDim[0];
        int njCurrent = current.blockInfo.totalDataDim[1];
        int nkCurrent = 1;
        int niNeighbor = neighbor.blockInfo.totalDataDim[0];
        int njNeighbor = neighbor.blockInfo.totalDataDim[1];
        int nkNeighbor = 1;
#if(CMF_IS3D)
        nkCurrent  = current.blockInfo.totalDataDim[2];
        nkNeighbor = neighbor.blockInfo.totalDataDim[2];
#endif
        bool exchangeDimsEqual = true;
        __dloop(exchangeDimsEqual = exchangeDimsEqual && (current.blockInfo.exchangeDim[d] == neighbor.blockInfo.exchangeDim[d]));
        
        //Create temporary arrays (these are not allocated)
        MdArray<char, 4> curArray(singleCellSize, niCurrent, njCurrent, nkCurrent);
        current.array = curArray;
        
        MdArray<char, 4> neiArray(singleCellSize, niCurrent, njCurrent, nkCurrent);
        neighbor.array = neiArray;
        
        //Manually assign data pointers
        current.array.data    = (char*)(meshArray->GetNodePointerWithNullDefault(currentNode));
        neighbor.array.data   = (char*)(meshArray->GetNodePointerWithNullDefault(neighborNode));
        
        Vec3<int> exchangeSizeCurrent(current.blockInfo.exchangeDim[0], current.blockInfo.exchangeDim[1], 0);
        Vec3<int> exchangeSizeNeighbor(neighbor.blockInfo.exchangeDim[0], neighbor.blockInfo.exchangeDim[1], 0);
        Vec3<int> meshSizeCurrent(niCurrent, njCurrent, nkCurrent);
        Vec3<int> meshSizeNeighbor(niNeighbor, njNeighbor, nkNeighbor);
#if(CMF_IS3D)
        exchangeSizeCurrent[2] = current.blockInfo.exchangeDim[2];
        exchangeSizeNeighbor[2] = neighbor.blockInfo.exchangeDim[2];
#endif

        current.exchangeSize = exchangeSizeCurrent;
        current.meshSize     = meshSizeCurrent;

        neighbor.exchangeSize = exchangeSizeNeighbor;
        neighbor.meshSize     = meshSizeNeighbor;
        
        //Check if the transaction will be a direct injection
        bool isDirectInjection = currentNode->IsSameDimensionsAs(neighborNode);        
        if (isDirectInjection)
        {
            CreateDirectInjectionTransaction(pattern, meshArray, current, neighbor, relationship);
        }
        else
        {
            CreateGeneralExchangePattern(pattern, meshArray, current, neighbor, relationship);
        }
    }
    
    void CartesianMeshExchangeHandler::CreateDirectInjectionTransaction
        (
            DataExchangePattern* pattern,
            CartesianMeshArray* meshArray,
            ExchangeContextBlockData& currentInfo,
            ExchangeContextBlockData& neighborInfo,
            NodeEdge& relationship
        )
    {
        size_t singleCellSize = currentInfo.array.dims[0];
        
        auto& currentData = currentInfo.array;
        auto& neighData = neighborInfo.array;
        
        //Check that all dimensions are equal among the blocks
        bool allDimsEqual = true;
        for (int d = 0; d < CMF_DIM; d++)
        {
            allDimsEqual = allDimsEqual && (currentInfo.exchangeSize[d]==neighborInfo.exchangeSize[d]);
            allDimsEqual = allDimsEqual && (currentInfo.meshSize[d]==neighborInfo.meshSize[d]);
        }

        if (!allDimsEqual)
        {
            std::string xonmeshy = "\"" + meshArray->variableName + "\" on mesh \"" + mesh->title + "\"";
            CmfError("Attempted to define a direct-injection pattern for variable " + xonmeshy + ", but found inconsistent dimensions");
        }
        
        Vec3<int> exchangeSize = currentInfo.exchangeSize;
        Vec3<int> meshSize = neighborInfo.meshSize - exchangeSize*2;
        
        //These will evetually determine the offsets
        Vec3<cell_t> ijkMinSend(0, 0, 0);
        Vec3<cell_t> ijkMaxSend(0, 0, 1); //fill with 1 in z position to catch 2D case
        Vec3<cell_t> ijkMinRecv(0, 0, 0);
        Vec3<cell_t> ijkMaxRecv(0, 0, 1);
        
        //IMPORTANT: each transaction is a single send by the current node and a receive from the neighbor!!
        for (int d = 0; d < CMF_DIM; d++)
        {
            //Get the bounding indices
            //Note: upper bounds are exclusive
            int edgeComponent = relationship.edgeVector[d];
            int num = meshSize[d];
            int exg = exchangeSize[d];
            switch (edgeComponent)
            {
                case -1: // sender is to the "right" of the receiver
                {
                    ijkMinSend[d] = exg;
                    ijkMaxSend[d] = ijkMinSend[d] + exg;
                    ijkMinRecv[d] = exg + num;
                    ijkMaxRecv[d] = ijkMinRecv[d] + exg;
                    break;
                }
                case 0: //sender "overlaps" receiver
                {
                    ijkMinSend[d] = exg;
                    ijkMaxSend[d] = ijkMinSend[d] + num;
                    ijkMinRecv[d] = exg;
                    ijkMaxRecv[d] = ijkMinRecv[d] + num;
                    break;
                }
                case 1: //sender is to the "left" of the receiver
                {
                    ijkMinSend[d] = num;
                    ijkMaxSend[d] = ijkMinSend[d] + exg;
                    ijkMinRecv[d] = 0;
                    ijkMaxRecv[d] = ijkMinRecv[d] + exg;
                    break;
                }
                default:
                {
                    CmfError("Attempted to create a direct injection transaction for var \"" + meshArray->variableName +"\" on mesh \""
                        + mesh->title +"\", but encountered invalid edge vector component \"" + std::to_string(edgeComponent) + "\"");
                    break;
                }
            }
        }        
        std::vector<size_t> offsetsSend;
        std::vector<size_t> sizesSend;
        for (cell_t k = ijkMinSend[2]; k < ijkMaxSend[2]; k++)
        {
            for (cell_t j = ijkMinSend[1]; j < ijkMaxSend[1]; j++)
            {
                cell_t imin = ijkMinSend[0];
                size_t packetSize = singleCellSize*(ijkMaxSend[0] - ijkMinSend[0]);
                size_t packetOffset = currentData.offset(0, imin, j, k);
                sizesSend.push_back(packetSize);
                offsetsSend.push_back(packetOffset);
            }
        }
        
        std::vector<size_t> offsetsRecv;
        std::vector<size_t> sizesRecv;
        for (cell_t k = ijkMinRecv[2]; k < ijkMaxRecv[2]; k++)
        {
            for (cell_t j = ijkMinRecv[1]; j < ijkMaxRecv[1]; j++)
            {
                cell_t imin = ijkMinRecv[0];
                size_t packetSize = singleCellSize*(ijkMaxRecv[0] - ijkMinRecv[0]);
                size_t packetOffset = neighData.offset(0, imin, j, k);
                sizesRecv.push_back(packetSize);
                offsetsRecv.push_back(packetOffset);
            }
        }
        
        int currentRank  = currentInfo.partitionInfo.rank;
        int neighborRank = neighborInfo.partitionInfo.rank;
        if (!currentInfo.partitionInfo.isCPU || !neighborInfo.partitionInfo.isCPU)
        {
            CmfError("GPU exchanges are not implemented yet!!!!");
        }
        
        //Get the relevant pointer. neither block is on the current rank, then the pointer is null,
        //but the transaction will immediately be deleted anyway.
        void* sendBuffer = (void*)(currentData.data);
        void* recvBuffer = (void*)(neighData.data);
        
        //Current node sends to the neighbor
        pattern->Add(new MultiTransaction(sendBuffer, offsetsSend, sizesSend, currentRank, recvBuffer, offsetsRecv, sizesRecv, neighborRank));
    }
    
    void CartesianMeshExchangeHandler::CreateGeneralExchangePattern
        (
            DataExchangePattern* pattern,
            CartesianMeshArray* meshArray,
            ExchangeContextBlockData& currentInfo,
            ExchangeContextBlockData& neighborInfo,
            NodeEdge& relationship
        )
    {
        Vec3<int> edgeVector(relationship.edgeVector[0], relationship.edgeVector[1], CMF_IS3D?0:relationship.edgeVector[CMF_DIM-1]);
        
        //the intersection of the ghost cells of the neighbor with the interior cells if the current form a rectangular prism
        //Interpreted in index-space coordinates
        Vec<double, 6> nonDimGhostOverlapRegion;
        for (int i = 0; i < 6; i++) nonDimGhostOverlapRegion[i] = 0;
        
        //refinement level of neighbor - refinement level of current
        // Vec3<int> neighborLevels = neighborInfo.node->GetDirectionLevels();
        // Vec3<int> currentLevels  = currentInfo.node->GetDirectionLevels();
        // Vec3<int> refineLevelDifference = neighborLevels - currentLevels;
        
        
    }
    
    CartesianMeshExchangeHandler::~CartesianMeshExchangeHandler(void)
    {
        for (auto p: exchanges)
        {
            delete p.second;
        }
    }
}