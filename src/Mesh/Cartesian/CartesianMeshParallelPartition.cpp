#include "CartesianMeshParallelPartition.h"
#include "CmfScreen.h"
#include "CartesianMesh.h"
#include "CmfError.h"
#include "ParallelGroup.h"
#include "BlockVtk.h"
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
            default:
            {
                CmfError("Invalid CartesianPartitionType type!");
                break;
            }
        }
        builder->CreatePartition(&partition, mesh);
    }
    
    void CartesianMeshParallelPartition::OnPostRefinementCallback(std::vector<RefinementTreeNode*>& newNodes)
    {
        WriteLine(3, "\"" + mesh->GetTitle() + "\" parallel partition handling new blocks");
        for (int i = 0; i < newNodes.size(); i++)
        {
            builder->AddNewNode(newNodes[i]);
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
                output["rank"] << (double)partition[lb.Node()].rank;
            }
            output.Write(filename);
        }
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
            return (partition[node].rank == mesh->meshGroup->Rank());
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