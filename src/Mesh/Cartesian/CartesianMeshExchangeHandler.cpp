#include "CartesianMeshExchangeHandler.h"
#include "CartesianMesh.h"
#include "RefinementConstraint.h"
#include "BlockIterator.h"
namespace cmf
{
    CartesianMeshExchangeHandler::CartesianMeshExchangeHandler(CartesianMesh* mesh_in, CartesianMeshExchangeInfo& inputInfo)
    {
        mesh = mesh_in;
        interpolationOrder = inputInfo.interpolationOrder;
        WriteLine(2, "Create exchange pattern on mesh \"" + mesh->GetTitle() + "\"");
        exchangeDim = inputInfo.exchangeDim;
        this->RegisterToBlocks(mesh->Blocks());
        if (mesh->Blocks()->GetRefinementConstraintType() != RefinementConstraint::factor2CompletelyConstrained)
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
        for (BlockIterator lb(meshArray, BlockFilters::Every, IterableMode::serial); lb.HasNext(); lb++)
        {
            
        }
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