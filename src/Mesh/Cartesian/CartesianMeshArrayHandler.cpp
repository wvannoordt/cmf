#include "CartesianMeshArrayHandler.h"
#include "CmfScreen.h"
#include "CartesianMesh.h"
#include "DebugTools.hx"
#include "CmfError.h"
namespace cmf
{
    CartesianMeshArrayHandler::CartesianMeshArrayHandler(CartesianMesh* mesh_in)
    {
        mesh = mesh_in;
        this->RegisterToBlocks(mesh_in->Blocks());
        defaultHandler = NULL;
        requireDeleteDefaultHandler = false;
    }
    
    CartesianMeshArray* CartesianMeshArrayHandler::CreateNewVariable(ArrayInfo info, NodeFilter_t filter)
    {
        if (VariableExists(info.name)) CmfError("Attempted to redefine variable \"" + info.name + "\" on mesh \"" + mesh->title + "\".");
        CartesianMeshArray* newArray = new CartesianMeshArray(info, this, filter);
        varList.insert({info.name, newArray});
        return newArray;
    }
    
    bool CartesianMeshArrayHandler::VariableExists(std::string name)
    {
        return (varList.find(name)!=varList.end());
    }
    
    CartesianMeshArray* CartesianMeshArrayHandler::GetVariable(std::string name)
    {
        if (!VariableExists(name)) CmfError("The mesh \"" + mesh->GetTitle() + "\" attempted to fetch a non-existent variable \"" + name + "\".");
        return (CartesianMeshArray*)varList[name];
    }
    
    void CartesianMeshArrayHandler::CreateExchangeHandler(CartesianMeshExchangeInfo& inputInfo)
    {
        requireDeleteDefaultHandler = true;
        defaultHandler = new CartesianMeshExchangeHandler(this->mesh, inputInfo);
    }
    
    CartesianMeshArrayHandler::~CartesianMeshArrayHandler(void)
    {
        if (requireDeleteDefaultHandler)
        {
            requireDeleteDefaultHandler = false;
            delete defaultHandler;
        }
    }
    
    void CartesianMeshArrayHandler::OnPostRefinementCallback(std::vector<RefinementTreeNode*>& newNodes)
    {
        // What do we do here? (restriction operator)
        if (varList.size()>0) CmfError("CartesianMeshArrayHandler::OnPostRefinementCallback is not implemented yet: cannot yet refine a mesh that contains a variable.");
    }
}