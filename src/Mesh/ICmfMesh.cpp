#include "ICmfMesh.h"
namespace cmf
{
    ICmfMesh::ICmfMesh(ICmfMeshInfo input)
    {
        title = input.title;
    }
    
    ICmfMesh::~ICmfMesh(void)
    {
        
    }
    
    ICmfMeshArrayHandler* ICmfMesh::GetArrayHandler(void)
    {
        return NULL;
    }
    
    void ICmfMesh::DefineVariable(std::string name)
    {
        
    }
}