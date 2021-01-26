#include "CartesianMeshExchangeHandler.h"
#include "CartesianMesh.h"
#include "RefinementConstraint.h"
#include "BlockIterator.h"
#include "Utils.hx"
#include "NeighborIterator.h"
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
        NodeFilter_t arFilter = meshArray->GetFilter();
        for (BlockIterator lb(meshArray, arFilter, IterableMode::serial); lb.HasNext(); lb++)
        {
            for (NeighborIterator neigh(lb.Node()); neigh.Active(); neigh++)
            {
                if (arFilter(neigh.Node()))
                {
                    RefinementTreeNode* currentNode  = lb.Node();
                    RefinementTreeNode* neighborNode = neigh.Node();
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
        BlockPartitionInfo currentNodeInfo  = mesh->partition->GetPartitionInfo(currentNode);
        BlockPartitionInfo neighborNodeInfo = mesh->partition->GetPartitionInfo(neighborNode);
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