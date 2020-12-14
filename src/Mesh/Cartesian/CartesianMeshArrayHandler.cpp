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
    
    CartesianMeshArrayHandler::~CartesianMeshArrayHandler(void)
    {
        
    }
}