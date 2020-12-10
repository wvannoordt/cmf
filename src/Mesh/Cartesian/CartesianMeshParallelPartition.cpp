#include "CartesianMeshParallelPartition.h"
#include "CmfScreen.h"
#include "CartesianMesh.h"
#include "CmfError.h"
#include "ParallelGroup.h"
namespace cmf
{
    CartesianMeshParallelPartition::CartesianMeshParallelPartition(CartesianMesh* mesh_in, CartesianMeshParallelPartitionInfo& inputInfo)
    {
        Build(mesh_in, inputInfo);
    }
    
    void CartesianMeshParallelPartition::Build(CartesianMesh* mesh_in, CartesianMeshParallelPartitionInfo& inputInfo)
    {
        mesh = mesh_in;
        partitionType = (CartesianPartitionType::CartesianPartitionType)inputInfo.partitionType;
        WriteLine(1, "Partition Cartesian mesh \"" + mesh->GetTitle()
            + "\" with strategy \"" + CartesianPartitionTypeStr(inputInfo.partitionType) + "\"");
        if (!mesh->meshGroup->IsInitialized())
        {
            CmfError("A partition for mesh \""+ mesh->GetTitle() + "\" was created without a valid parallel context!");
        }
    }
    
    CartesianMeshParallelPartition::~CartesianMeshParallelPartition(void)
    {
        
    }
}