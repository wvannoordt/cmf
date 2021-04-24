#include "ICmfMesh.h"
#include "CmfPrint.h"
namespace cmf
{
    ICmfMesh::ICmfMesh(ICmfMeshInfo input, MeshType::MeshType meshType_in)
    {
        title = input.title;
        meshType = meshType_in;
    }
    
    std::string ICmfMesh::GetTitle(void)
    {
        return title;
    }
    
    ICmfMesh::~ICmfMesh(void)
    {
        
    }
}