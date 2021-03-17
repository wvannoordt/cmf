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
            /// @param sendTarget_in The target address for the sender, or NULL if the receiving array is on another rank
            /// @param sendOffsets_in A list of offsets to copy from, relative to sendTarget_in. Ignored if sendRank_in is not rhe current rank
            /// @param sendSizes_in The size of each copy in the corresponding offset. Ignored if sendRank_in is not rhe current rank
            /// @param sendRank_in The rank of the sender
            /// @param recvTarget_in The target address for the receiver, or NULL if the receiving array is on another rank
            /// @param recvOffsets_in A list of offsets to copy from, relative to receiveTarget_in. Ignored if recvRank_in is not rhe current rank
            /// @param recvSizes_in The size of each copy in the corresponding offset. Ignored if recvRank_in is not rhe current rank
            /// @param recvRank_in The rank of the receiver
            /// @author WVN
            MultiTransaction(
                void* sendTarget_in, std::vector<size_t> sendOffsets_in, std::vector<size_t> sendSizes_in, int sendRank_in,
                void* recvTarget_in, std::vector<size_t> recvOffsets_in, std::vector<size_t> recvSizes_in, int recvRank_in);
            
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
            
            /// @brief The address of the array to be sent, or NULL if it lies on another rank
            void* sendTarget;
            
            /// @brief The offsets (relative to sendTarget) to send. Ignored if sendRank is not rhe current rank
            std::vector<size_t> sendOffsets;
            
            /// @brief The sizes to send. Ignored if sendRank is not rhe current rank
            std::vector<size_t> sendSizes;
            
            /// @brief The sending rank
            int sendRank;
            
            /// @brief The address of the array to be received, or NULL if it lies on another rank
            void* recvTarget;
            
            /// @brief The offsets (relative to recvTarget) to receive. Ignored if recvRank is not rhe current rank
            std::vector<size_t> recvOffsets;
            
            /// @brief The sizes to receive. Ignored if recvRank is not rhe current rank
            std::vector<size_t> recvSizes;
            
            /// @brief The receiving rank
            int recvRank;
            
    };
}

#endif