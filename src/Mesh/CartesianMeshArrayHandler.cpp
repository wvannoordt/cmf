#include "CartesianMeshArrayHandler.h"
#include "CmfScreen.h"
#include "DebugTools.hx"
#include "CmfError.h"
namespace cmf
{
    CartesianMeshArrayHandler::CartesianMeshArrayHandler(void)
    {
        
    }
    
    void CartesianMeshArrayHandler::CreateNewVariable(ArrayInfo info)
    {
        if (VariableExists(info.name)) CmfError("Attempted to redefine variable \"" + info.name + "\".");
        varList.insert({info.name, new CartesianMeshArray(info)});
    }
    
    
    
    bool CartesianMeshArrayHandler::VariableExists(std::string name)
    {
        return (varList.find(name)!=varList.end());
    }
    
    CartesianMeshArrayHandler::~CartesianMeshArrayHandler(void)
    {
        
    }
}