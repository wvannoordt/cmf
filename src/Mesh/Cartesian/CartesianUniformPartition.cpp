#include "CartesianUniformPartition.h"
#include "CartesianMesh.h"
#include "CmfScreen.h"
#include "BlockIterator.h"
#include "BlockFilters.h"
#include "CmfPrint.h"
namespace cmf
{
    CartesianUniformPartition::CartesianUniformPartition(void)
    {
        counter = -1;
        numRanks = -1;
    }
    
    CartesianUniformPartition::~CartesianUniformPartition(void)
    {
        
    }
    
    void CartesianUniformPartition::CreatePartition(std::map<RefinementTreeNode*, ComputeDevice>* partition, CartesianMesh* mesh)
    {
        meshPartition = partition;
        counter = 0;
        numRanks = mesh->GetGroup()->Size();
        //Note that every single block, regardless of status, must be partitioned, hence the serial loop
        for (BlockIterator lb(mesh, BlockFilters::Every, IterableMode::serial); lb.HasNext(); lb++)
        {
            AddNewNode(lb.Node());
        }
    }
    
    void CartesianUniformPartition::AddNewNode(RefinementTreeNode* newNode)
    {
        ComputeDevice newDevice;
        newDevice.id = counter;
        newDevice.isGpu = false;
        meshPartition->insert({newNode, newDevice});
        IncrementCounter();
    }
    
    void CartesianUniformPartition::IncrementCounter(void)
    {
        counter++;
        counter = counter % numRanks;
    }
}