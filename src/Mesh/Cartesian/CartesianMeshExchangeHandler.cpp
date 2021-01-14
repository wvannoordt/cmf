#include "CartesianMeshExchangeHandler.h"
#include "CartesianMesh.h"
#include "RefinementConstraint.h"
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
    
    void CartesianMeshExchangeHandler::OnPostRefinementCallback(std::vector<RefinementTreeNode*>& newNodes)
    {
        WriteLine(1, "WARNING!!!!!!! CartesianMeshExchangeHandler::OnPostRefinementCallback not implemented");
    }
    
    CartesianMeshExchangeHandler::~CartesianMeshExchangeHandler(void)
    {
        
    }
}