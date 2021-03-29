#include "ICmfMesh.h"
#include "CmfPrint.h"
namespace cmf
{
    ICmfMesh::ICmfMesh(ICmfMeshInfo input, MeshType::MeshType meshType_in)
    {
        title = input.title;
        meshType = meshType_in;
    }
    
    std::string ICmfMesh::DataBaseName(void)
    {
        return title;
    }
    
    std::string ICmfMesh::GetTitle(void)
    {
        return title;
    }
    
    void ICmfMesh::SetRequiredPrereqtuisiteDataBaseObjects(void)
    {
        
    }

    void ICmfMesh::SetAutomaticallyAddedObjects(void)
    {
        objectsToAutomaticallyAddWhenAddingToDataBase.Add(baseMeshArrayHandler);
    }
    
    ICmfMesh::~ICmfMesh(void)
    {
        
    }
}