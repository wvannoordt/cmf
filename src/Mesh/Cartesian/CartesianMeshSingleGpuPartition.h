#ifndef CMF_CARTESIAN_SINGLE_GPU_PARTITION_H
#define CMF_CARTESIAN_SINGLE_GPU_PARTITION_H
#include "ICartesianPartitionBuilder.h"
namespace cmf
{
    ///@brief Represents a homogeneous partition over a single GPU
    ///@author WVN
    class CartesianMeshSingleGpuPartition : public ICartesianPartitionBuilder
    {
        public:
            ///@brief Empty constructor
            ///@author WVN
            CartesianMeshSingleGpuPartition(void);
            
            /// @brief Builds the given partition over the given mesh
            /// @param partition The partition to populate
            /// @param mesh The mesh containing the blocks to be partitioned
            /// @author WVN
            virtual void CreatePartition(std::map<RefinementTreeNode*, ComputeDevice>* partition, CartesianMesh* mesh) override final;
            
            /// @brief Adds a new node to the partition and assigns a rank according to the specialized strategy
            /// @param newNode The new node to add
            /// @author WVN
            virtual void AddNewNode(RefinementTreeNode* newNode) override final;
    };
}

#endif