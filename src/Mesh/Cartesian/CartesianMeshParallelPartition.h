#ifndef CMF_CARTESIAN_PARALLEL_PARTITION_INFO_H
#define CMF_CARTESIAN_PARALLEL_PARTITION_INFO_H
#include "ICmfInputObject.h"
#include "CartesianPartitionType.h"
#include "RefinementTreeNode.h"
#include "ICartesianPartitionBuilder.h"
#include "CartesianUniformPartition.h"
#include <map>
namespace cmf
{
    
    class CartesianMesh;
    /// @brief A struct containing all the input information for a parallel-hybrid partition over a Cartesian mesh.
    /// In short, This object will iterate over all blocks in the provided mesh and will assign an MPI rank.
    /// @author WVN
    struct CartesianMeshParallelPartitionInfo : ICmfInputObject
    {
        /// @brief The type of partition
        int partitionType;
        
        /// @brief Constructor for the CartesianMeshInputInfo object.
        /// @param inputSection Section to be read from
        /// @author WVN
        CartesianMeshParallelPartitionInfo(PropTreeLib::PropertySection& inputSection) : ICmfInputObject(inputSection)
        {
            Define(*objectInput);
            Parse();
        }
        
        /// @brief Defines the object from the input secton
        /// @param input The section to be read from
        /// @author WVN
        void Define(PropTreeLib::PropertySection& input)
        {
            input["partitionType"].MapTo(&partitionType)
                = new PropTreeLib::Variables::PTLAutoEnum(CartesianPartitionType::uniform, CartesianPartitionTypeStr, "The partitioning approach used to partition the Cartesian mesh");
        }
    };
    
    /// @brief A class defining a parallel-hybrid partition over a cartesian mesh
    /// @author WVN
    class CartesianMeshParallelPartition
    {
        public:
            /// @brief Constructor for CartesianMeshParallelPartition
            /// @param mesh_in The mesh to be partitioned
            /// @param inputInfo The struct containing input parameters for constructing this object
            /// @author WVN
            CartesianMeshParallelPartition(CartesianMesh* mesh_in, CartesianMeshParallelPartitionInfo& inputInfo);
            
            /// @brief Destructor
            /// @author WVN
            ~CartesianMeshParallelPartition(void);
            
            /// @brief Returns true if the current rank is responsible for the given node
            /// @param node The node to evaluate
            /// @author WVN
            bool Mine(RefinementTreeNode* node);
            
        private:
            
            /// @brief Translates a CartesianMeshParallelPartitionInfo to the information 
            /// @param inputInfo The struct containing input parameters for constructing this object
            /// @author WVN
            void Build(CartesianMesh* mesh_in, CartesianMeshParallelPartitionInfo& inputInfo);
            
            /// @brief The mesh that this object partitions
            CartesianMesh* mesh;
            
            /// @brief The type of partition used to partition the mesh
            CartesianPartitionType::CartesianPartitionType partitionType;
            
            /// @brief The parallel partition data
            std::map<RefinementTreeNode*, int> partition;
            
            /// @brief The object responsible for assigning existing and new blocks to the parallel partition
            ICartesianPartitionBuilder* builder;
            
            /// @brief Indicates whether or not builder must be deleted
            bool deleteBuilder;
    };
}

#endif