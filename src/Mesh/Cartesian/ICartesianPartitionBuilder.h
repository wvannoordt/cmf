#ifndef CMT_CART_MESH_PAR_PART_BUI_H
#define CMT_CART_MESH_PAR_PART_BUI_H
#include "RefinementTreeNode.h"
#include <map>
namespace cmf
{
    class CartesianMesh;
    /// @brief A base class to be inherited by specialied classes used for building the parallel partitions
    /// @author WVN
    class ICartesianPartitionBuilder
    {
        public:
            /// @brief Constructor
            /// @author WVN
            ICartesianPartitionBuilder(void);
            
            /// @brief Destructor
            /// @author WVN
            ~ICartesianPartitionBuilder(void);
            
            /// @brief Builds the given partition over the given mesh
            /// @param partition The partition to populate
            /// @param mesh The mesh containing the blocks to be partitioned
            /// @author WVN
            virtual void CreatePartition(std::map<RefinementTreeNode*, int>& partition, CartesianMesh* mesh) = 0;
            
    };
}

#endif