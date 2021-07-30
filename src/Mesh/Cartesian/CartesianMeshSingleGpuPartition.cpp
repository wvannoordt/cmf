#include "CartesianMeshSingleGpuPartition.h"
#include "CartesianMesh.h"
#include "CmfError.h"
namespace cmf
{
    CartesianMeshSingleGpuPartition::CartesianMeshSingleGpuPartition()
    {
        
    }
    
    void CartesianMeshSingleGpuPartition::CreatePartition(std::map<RefinementTreeNode*, ComputeDevice>* partition, CartesianMesh* mesh)
    {
        meshPartition = partition;
        auto& p = *partition;
        if (mesh->GetGroup()->Size()>1)
        {
            CmfError("Attempted to use a homogeneous GPU partition when running in parallel on the CPU. This is not supported.");
        }
#if (!CUDA_ENABLE)
            CmfError("Attempted to use a homogeneous GPU partition without CUDA support. This will not work.");
#endif
        for (BlockIterator lb(mesh, BlockFilters::Every, IterableMode::serial); lb.HasNext(); lb++)
        {
            this->AddNewNode(lb.Node());
        }
    }
    
    void CartesianMeshSingleGpuPartition::AddNewNode(RefinementTreeNode* newNode)
    {
        ComputeDevice newDevice;
        newDevice.id = 0;
        newDevice.isGpu = true;
        meshPartition->insert({newNode, newDevice});
    }
}