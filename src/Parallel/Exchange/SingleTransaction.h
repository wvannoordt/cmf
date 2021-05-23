#ifndef CMF_SINGLE_TRANSACTION_H
#define CMF_SINGLE_TRANSACTION_H
#include "IDataTransaction.h"
namespace cmf
{
    /// @brief A class representing a data exchange transaction of a single, contiguous memory buffer
    /// @author WVN
    class SingleTransaction : public IDataTransaction
    {
        public:
            /// @brief Constructor
            /// @param sendTarget_in The target address for the sender, or NULL if the receiving array is on another rank
            /// @param recvTarget_in The target address for the receiver, or NULL if the receiving array is on another rank
            /// @param size_in The size of the transaction
            /// @param sender_in The sending rank
            /// @param receiver_in The receiving rank
            /// @author WVN
            SingleTransaction(void* sendTarget_in, void* recvTarget_in, size_t size_in, int sender_in, int receiver_in);
            
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
            
            /// @brief Destructor
            /// @author WVN
            ~SingleTransaction(void);
            
        private:
            
            /// @brief The address of the array to be sent, or NULL if it lies on another rank
            void* sendTarget;
            
            /// @brief The address of the array to be received, or NULL if it lies on another rank
            void* recvTarget;
            
            /// brief The size of the transaction
            size_t size;
            
            
    };
}

#endif