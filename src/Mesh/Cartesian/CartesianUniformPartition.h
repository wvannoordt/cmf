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
            void CreatePartition(std::map<RefinementTreeNode*, int>& partition, CartesianMesh* mesh);
    };
}

#endif