#ifndef CMF_MULTI_TRANSACTION
#define CMF_MULTI_TRANSACTION
#include "CmfGC.h"
#include "IDataTransaction.h"
#include <vector>

namespace cmf
{
    /// @brief A class representing a data exchange transaction multiple contiguous regions at various offsets from an address
    /// @author WVN
    class MultiTransaction : public IDataTransaction
    {
        public:
            /// @brief Constructor
            /// @param target_in The target address
            /// @param offsets_in A list of offsets to copy from, relative to target_in
            /// @param sizes_in The size of each copy in the corresponding offset
            /// @param sender_in The sending rank
            /// @param receiver_in The receiving rank
            /// @author WVN
            MultiTransaction(void* target_in, std::vector<size_t> offsets_in, std::vector<size_t> sizes_in, int sender_in, int receiver_in);
            
            /// @brief Destructor
            /// @author WVN
            ~MultiTransaction(void);
            
            /// @brief Returns the size of the compacted data
            /// @author WVN
            size_t GetPackedSize(void) override;
            
            /// @brief Packs the data to the given buffer
            /// @param buf The buffer to pack the data to
            /// \pre Note that the size of buf must be at least the size returned by GetPackedSize()
            /// @author WVN
            void Pack(char* buf) override;
            
            /// @brief Unpacks the data from the given buffer
            /// @param buf The buffer to unpack the data from
            /// \pre Note that the size of buf must be at least the size returned by GetPackedSize()
            /// @author WVN
            void Unpack(char* buf) override;
            
        private:
            
            /// @brief The total size of the data transaction
            size_t packedSize;
            
            /// @brief The target address for this transaction
            void* target;
            
            /// @brief The list of offsets to copy from/to, relative to target
            std::vector<size_t> offsets;
            
            /// @brief The list of sizes of each copy
            std::vector<size_t> sizes;
    };
}

#endif