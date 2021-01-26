#ifndef CMF_BLOCK_PARTITION_INFO_H
#define CMF_BLOCK_PARTITION_INFO_H

namespace cmf
{
    /// @brief A struct containing partition info for a computational block
    /// @author WVN
    struct BlockPartitionInfo
    {
        /// @brief Default constructor
        /// @author WVN
        BlockPartitionInfo(void)
        {
            isCPU = true;
            rank = 0;
        }
        
        /// @brief Indicates whether or not the associated block is handled by the CPU
        bool isCPU;
        
        /// @brief The MPI rank responsible for the associated block
        int rank;
    };
}

#endif