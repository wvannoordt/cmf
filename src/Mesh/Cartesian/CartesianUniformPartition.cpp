#include "CartesianUniformPartition.h"
#include "CartesianMesh.h"
#include "CmfScreen.h"
namespace cmf
{
    CartesianUniformPartition::CartesianUniformPartition(void)
    {
        
    }
    
    CartesianUniformPartition::~CartesianUniformPartition(void)
    {
        
    }
    
    void CartesianUniformPartition::CreatePartition(std::map<RefinementTreeNode*, int>& partition, CartesianMesh* mesh)
    {
        WriteLine(1, "TEMPORARY MESSAGE");
    }
}