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
        deleteBuilder = false;
        mesh = mesh_in;
        partitionType = (CartesianPartitionType::CartesianPartitionType)inputInfo.partitionType;
        WriteLine(1, "Partition Cartesian mesh \"" + mesh->GetTitle()
            + "\" with strategy \"" + CartesianPartitionTypeStr(inputInfo.partitionType) + "\"");
        if (!mesh->meshGroup->IsInitialized())
        {
            CmfError("A partition for mesh \""+ mesh->GetTitle() + "\" was created without a valid parallel context!");
        }
        switch(partitionType)
        {
            case CartesianPartitionType::uniform:
            {
                deleteBuilder = true;
                builder = new CartesianUniformPartition();                
                break;
            }
        }
        builder->CreatePartition(partition, mesh);
    }
    
    bool CartesianMeshParallelPartition::Mine(RefinementTreeNode* node)
    {
        if (partition.find(node) == partition.end())
        {
            CmfError("Attempted to fetch a non-existent node on Cartesian mesh \"" + mesh->GetTitle() + "\".");
            return false;
        }
        else
        {
            return (partition[node] == mesh->meshGroup->Rank());
        }
    }
    
    CartesianMeshParallelPartition::~CartesianMeshParallelPartition(void)
    {
        if (deleteBuilder)
        {
            deleteBuilder = false;
            delete builder;
        }
    }
}