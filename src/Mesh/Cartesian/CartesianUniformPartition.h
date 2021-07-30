#ifndef CMF_CART_UNIFORM_PART_H
#define CMF_CART_UNIFORM_PART_H
#include "ICartesianPartitionBuilder.h"
namespace cmf
{
    /// @brief A class that implements a uniform partitioning strategy
    /// @author WVN
    class CartesianUniformPartition : public ICartesianPartitionBuilder
    {
        public:
            /// @brief Constructor
            /// @author WVN
            CartesianUniformPartition(void);
            
            /// @brief Destructor
            /// @author WVN
            ~CartesianUniformPartition(void);
            
            /// @brief Builds the given partition over the given mesh
            /// @param partition The partition to populate
            /// @param mesh The mesh containing the blocks to be partitioned
            /// @author WVN
            void CreatePartition(std::map<RefinementTreeNode*, ComputeDevice>* partition, CartesianMesh* mesh);
            
            /// @brief Adds a new node to the partition and assigns a rank according to the specialized strategy
            /// @param newNode The new node to add
            /// @author WVN
            void AddNewNode(RefinementTreeNode* newNode);
        
        private:
            
            /// @brief Increments the counter
            void IncrementCounter(void);
            
            /// @brief A counter incremented each time a block is added. Indicates the next ranks to be assigned to
            int counter;
            
            /// @brief The total number of ranks available to assign to
            int numRanks;
    };
}

#endif