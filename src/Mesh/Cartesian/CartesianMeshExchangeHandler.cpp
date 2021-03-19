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
        this->RegisterToBlocks(mesh->Blocks());
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
            CmfError("Attempted to define duplicate exchange pattern for array \"" + meshArray->GetFullName() + "\"");
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
                    //This might need to become more sophisticated
                    bool isDirectInjection = currentNode->IsSameDimensionsAs(neighborNode);
                    if (isDirectInjection)
                    {
                        CreateDirectInjectionTransaction(pattern, meshArray, currentNode, neighborNode, neigh.Edge());
                    }
                    else
                    {
                        WriteLine(1, "WARNING: defining exchange patterns not yet implemented between different refinement levels");
                    }
                }
            }
        }
    }
    
    void CartesianMeshExchangeHandler::CreateDirectInjectionTransaction(
        DataExchangePattern* pattern,
        CartesianMeshArray* meshArray,
        RefinementTreeNode* currentNode,
        RefinementTreeNode* neighborNode, 
        NodeEdge relationship)
    {
        //This almost certainly needs to be broken up into sub-functions...
        
        //Retrieve partition info and block dimensions from the mesh
        BlockPartitionInfo currentNodeInfo  = mesh->partition->GetPartitionInfo(currentNode);
        BlockPartitionInfo neighborNodeInfo = mesh->partition->GetPartitionInfo(neighborNode);
        BlockInfo currentNodeBlockInfo  = mesh->GetBlockInfo(currentNode);
        BlockInfo neighborNodeBlockInfo = mesh->GetBlockInfo(neighborNode);
        
        //Compute the size of a single array "element"
        size_t singleCellSize = meshArray->elementSize;
        for (int i = 0; i < meshArray->rank; i++)
        {
            singleCellSize *= meshArray->dims[i];
        }
        //Compute array sizes for data transfer
        int niCurrent = currentNodeBlockInfo.totalDataDim[0];
        int njCurrent = currentNodeBlockInfo.totalDataDim[1];
        int nkCurrent = 1;
        int niNeighbor = neighborNodeBlockInfo.totalDataDim[0];
        int njNeighbor = neighborNodeBlockInfo.totalDataDim[1];
        int nkNeighbor = 1;
#if(CMF_IS3D)
        nkCurrent  = currentNodeBlockInfo.totalDataDim[2];
        nkNeighbor = neighborNodeBlockInfo.totalDataDim[2];
#endif
        bool exchangeDimsEqual = true;
        __dloop(exchangeDimsEqual = exchangeDimsEqual && (currentNodeBlockInfo.exchangeDim[d] == neighborNodeBlockInfo.exchangeDim[d]));
        
        //Create temporary arrays (these are not allocated)
        MdArray<char, 4> currentData(singleCellSize, niCurrent, njCurrent, nkCurrent);
        MdArray<char, 4> neighData(singleCellSize, niNeighbor, njNeighbor, nkNeighbor);
        
        //Manually assign data pointers
        currentData.data = (char*)(meshArray->GetNodePointerWithNullDefault(currentNode));
        neighData.data   = (char*)(meshArray->GetNodePointerWithNullDefault(neighborNode));
        
        //Check that all dimensions are equal among the blocks
        bool allDimsEqual = true;
        allDimsEqual = allDimsEqual && (niNeighbor==niCurrent);
        allDimsEqual = allDimsEqual && (njNeighbor==njCurrent);
        allDimsEqual = allDimsEqual && (nkNeighbor==nkCurrent);
        
        Vec3<int> exchangeSizeCurrent(currentNodeBlockInfo.exchangeDim[0], currentNodeBlockInfo.exchangeDim[1], 0);
        Vec3<int> exchangeSizeNeighbor(neighborNodeBlockInfo.exchangeDim[0], neighborNodeBlockInfo.exchangeDim[1], 0);
        Vec3<int> meshSizeCurrent(niCurrent, njCurrent, nkCurrent);
        Vec3<int> meshSizeNeighbor(niNeighbor, njNeighbor, nkNeighbor);
#if(CMF_IS3D)
        exchangeSizeCurrent[2] = currentNodeBlockInfo.exchangeDim[2];
        exchangeSizeNeighbor[2] = neighborNodeBlockInfo.exchangeDim[2];
#endif
        Vec3<int> exchangeSize = exchangeSizeNeighbor;
        Vec3<int> meshSize = meshSizeCurrent - exchangeSize*2;

        if (!exchangeDimsEqual || !allDimsEqual)
        {
            std::string xonmeshy = "\"" + meshArray->variableName + "\" on mesh \"" + mesh->title + "\"";
            CmfError("Attempted to define a direct-injection pattern for variable " + xonmeshy + ", but found inconsistent dimensions: \nBlock dimensions:    " + 
                meshSizeCurrent.str() + " / " + meshSizeNeighbor.str() + "\nExchange dimensions: " + exchangeSizeCurrent.str() + " / " + exchangeSizeNeighbor.str());
        }
        
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
        
        int currentRank  = currentNodeInfo.rank;
        int neighborRank = neighborNodeInfo.rank;
        if (!currentNodeInfo.isCPU || !neighborNodeInfo.isCPU)
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
    
    void CartesianMeshExchangeHandler::OnPostRefinementCallback(std::vector<RefinementTreeNode*>& newNodes)
    {
        WriteLine(1, "WARNING!!!!!!! CartesianMeshExchangeHandler::OnPostRefinementCallback not implemented");
    }
    
    CartesianMeshExchangeHandler::~CartesianMeshExchangeHandler(void)
    {
        for (auto p: exchanges)
        {
            delete p.second;
        }
    }
}