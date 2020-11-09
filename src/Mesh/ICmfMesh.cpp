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
    
    //ICmfMeshArray* ICmfMesh::DefineVariable(std::string name)
    //{
//        return NULL;
//    }
}