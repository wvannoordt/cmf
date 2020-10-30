#include "ICmfMeshArrayHandler.h"
#include "ICmfMesh.h"
namespace cmf
{
    ICmfMeshArrayHandler::ICmfMeshArrayHandler(void)
    {
        
    }
    
    ICmfMeshArrayHandler::~ICmfMeshArrayHandler(void)
    {
        Destroy();
    }
    
    void ICmfMeshArrayHandler::CreateNewVariable(ArrayInfo info)
    {
        
    }
    
    bool ICmfMeshArrayHandler::VariableExists(std::string name)
    {
        return (varList.find(name)!=varList.end());
    }
    
    void ICmfMeshArrayHandler::Destroy(void)
    {
        for (std::map<std::string, ICmfMeshArray*>::iterator it = varList.begin(); it != varList.end(); it++)
        {
            WriteLine(6, "Free var \"" + it->first + "\"");
            it->second->Destroy();
            delete it->second;
        }
    }
}