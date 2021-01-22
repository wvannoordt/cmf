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
            /// @param target_in The target address
            /// @param size_in The size of the transaction
            /// @param sender_in The sending rank
            /// @param receiver_in The receiving rank
            /// @author WVN
            SingleTransaction(void* target_in, size_t size_in, int sender_in, int receiver_in);
            
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
            
            /// @brief Constructor
            /// @author WVN
            ~SingleTransaction(void);
            
        private:
            
            /// brief The target address
            void* target;
            
            /// brief The size of the transaction
            size_t size;
            
            
    };
}

#endif