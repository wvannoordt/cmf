#ifndef CMF_CARTESIAN_PARALLEL_PARTITION_INFO_H
#define CMF_CARTESIAN_PARALLEL_PARTITION_INFO_H
#include "ICmfInputObject.h"
#include "CartesianPartitionType.h"
#include "RefinementTreeNode.h"
#include "ICartesianPartitionBuilder.h"
#include "CartesianUniformPartition.h"
#include "BlockPartitionInfo.h"
#include "IPostRefinementCallback.h"
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
        
        /// @brief empty Constructor
        /// @author WVN
        CartesianMeshParallelPartitionInfo(void){}
        
        /// @brief Constructor for the CartesianMeshInputInfo object.
        /// @param inputSection Section to be read from
        /// @author WVN
        CartesianMeshParallelPartitionInfo(PTL::PropertySection& inputSection) : ICmfInputObject(inputSection)
        {
            Define(*objectInput);
            Parse();
        }
        
        /// @brief Defines the object from the input secton
        /// @param input The section to be read from
        /// @author WVN
        void Define(PTL::PropertySection& input)
        {
            input["partitionType"].MapTo(&partitionType)
                = new PTL::PTLAutoEnum(CartesianPartitionType::uniform, CartesianPartitionTypeStr, "The partitioning approach used to partition the Cartesian mesh");
        }
    };
    
    /// @brief A class defining a parallel-hybrid partition over a cartesian mesh
    /// @author WVN
    class CartesianMeshParallelPartition : public IPostRefinementCallback
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
            
            /// @brief The callback function for new nodes
            /// @param newNodes The newly refined nodes to be handled
            /// @author WVN
            void OnPostRefinementCallback(std::vector<RefinementTreeNode*>& newNodes);
            
            /// @brief Outputs the parallel partition to a VTK file
            /// @param filename The file to be written to
            /// @author WVN
            void OutputPartitionToVtk(std::string filename);
            
            /// @brief Returns the BlockPartitionInfo for the given node
            /// @param node The node to get the partition info for
            /// @author WVN
            BlockPartitionInfo GetPartitionInfo(RefinementTreeNode* node);
            
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
            std::map<RefinementTreeNode*, BlockPartitionInfo> partition;
            
            /// @brief The object responsible for assigning existing and new blocks to the parallel partition
            ICartesianPartitionBuilder* builder;
            
            /// @brief Indicates whether or not builder must be deleted
            bool deleteBuilder;
    };
}

#endif