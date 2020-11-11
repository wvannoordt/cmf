#include "ICmfMesh.h"
namespace cmf
{
    ICmfMesh::ICmfMesh(ICmfMeshInfo input, MeshType::MeshType meshType_in)
    {
        title = input.title;
        meshType = meshType_in;
    }
    
    ICmfMesh::~ICmfMesh(void)
    {
        
    }
    
    ICmfMeshArrayHandler* ICmfMesh::GetArrayHandler(void)
    {
        return NULL;
    }
}