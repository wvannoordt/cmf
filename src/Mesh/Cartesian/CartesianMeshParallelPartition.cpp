#include "CartesianMeshParallelPartition.h"
#include "CmfScreen.h"
#include "CartesianMesh.h"
#include "CmfError.h"
#include "ParallelGroup.h"
#include "BlockVtk.h"
#include "StringUtils.h"
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
        RegisterToBlocks(mesh->Blocks());
        WriteLine(1, "Partition Cartesian mesh \"" + mesh->GetTitle()
            + "\" with strategy \"" + CartesianPartitionTypeStr(inputInfo.partitionType) + "\"");
        switch(partitionType)
        {
            case CartesianPartitionType::uniform:
            {
                deleteBuilder = true;
                builder = new CartesianUniformPartition();                
                break;
            }
            case CartesianPartitionType::singleGPU:
            {
                deleteBuilder = true;
                builder = new CartesianMeshSingleGpuPartition();                
                break;
            }
            default:
            {
                CmfError("Invalid CartesianPartitionType type!");
                break;
            }
        }
        builder->CreatePartition(&partition, mesh);
    }
    
    ComputeDevice CartesianMeshParallelPartition::GetPartitionInfo(RefinementTreeNode* node)
    {
        if (partition.find(node)==partition.end()) CmfError("Requested partition for an unpartitioned node.");
        return partition[node];
    }
    
    void CartesianMeshParallelPartition::OnPostRefinementCallback(std::vector<RefinementTreeNode*>& newChildNodes, std::vector<RefinementTreeNode*> newParentNodes)
    {
        WriteLine(4, "\"" + mesh->GetTitle() + "\" parallel partition handling new blocks");
        for (int i = 0; i < newChildNodes.size(); i++)
        {
            builder->AddNewNode(newChildNodes[i]);
        }
        mesh->AssertSynchronizeBlocks();
    }
    
    void CartesianMeshParallelPartition::OutputPartitionToVtk(std::string filename)
    {
        if (mesh->GetGroup()->IsRoot())
        {
            BlockVtk output;
            for (BlockIterator lb(mesh, BlockFilters::Terminal, IterableMode::serial); lb.HasNext(); lb++)
            {
                output << lb.Node();
                output["rank"] << (double)(partition[lb.Node()].id - partition[lb.Node()].isGpu?1:0)*(partition[lb.Node()].isGpu?-1:1);
            }
            output.Write(filename);
        }
    }
    
    bool CartesianMeshParallelPartition::Mine(RefinementTreeNode* node)
    {
        if (partition.find(node) == partition.end())
        {
            CmfError(strformat("Attempted to fetch a non-existent node {} on Cartesian mesh \"{}\"", node, mesh->GetTitle()));
            return false;
        }
        else
        {
            return (partition[node].id == mesh->meshGroup->Rank().id);
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