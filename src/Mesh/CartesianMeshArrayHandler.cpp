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
    
    void CartesianMeshArrayHandler::CreateNewVariable(ArrayInfo info)
    {
        if (VariableExists(info.name)) CmfError("Attempted to redefine variable \"" + info.name + "\" on mesh \"" + mesh->title + "\".");
        varList.insert({info.name, new CartesianMeshArray(info, this)});
    }
    
    
    
    bool CartesianMeshArrayHandler::VariableExists(std::string name)
    {
        return (varList.find(name)!=varList.end());
    }
    
    CartesianMeshArrayHandler::~CartesianMeshArrayHandler(void)
    {
        
    }
}