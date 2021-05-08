#ifndef CMF_EXCHANGE_CONTEXT_BLOCK_DATA_H
#define CMF_EXCHANGE_CONTEXT_BLOCK_DATA_H
#include "Vec.h"
#include "RefinementTreeNode.h"
#include "BlockPartitionInfo.h"
#include "BlockInfo.h"
#include "BlockArray.h"


namespace cmf
{
    /// @brief Holds all necessary information about a block
    /// in the context of defining an exchange pattern
    /// @author WVN
    struct ExchangeContextBlockData
    {
        /// @brief the node that this data pertains to
        RefinementTreeNode* node;
        
        /// @brief contains information about the partition for this block
        BlockPartitionInfo  partitionInfo;
        
        /// @brief contains mesh information about this block
        BlockInfo           blockInfo;
        
        /// @brief The data array for this block
        MdArray<char, 4>    array;
        
        /// @brief the size of the exchange cells, third component is 1 in 2-D
        Vec3<int>           exchangeSize;
        
        /// @brief the size of the mesh cells (including exchanges),
        /// third component is 1 in 2-D
        Vec3<int>           meshSize;
    };
}

#endif