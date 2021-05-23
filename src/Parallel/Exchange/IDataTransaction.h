#ifndef CMF_IDATA_TRANSACTION_H
#define CMF_IDATA_TRANSACTION_H

#include "CmfGC.h"

namespace cmf
{
    /// @brief Defines a generic parallel data transaction (including self-to-self transactions)
    /// \pre Note that these objects are intended to exist independently of parallel groups, and are tied
    /// to them via the DataExchangePattern class.
	/// @author WVN
    class IDataTransaction
    {
        public:
            /// @brief Constructor
            /// @param sender_in The sending rank
            /// @param receiver_in The receiving rank
        	/// @author WVN
            IDataTransaction(int sender_in, int receiver_in);
            
            /// @brief Destructor
        	/// @author WVN
            virtual ~IDataTransaction(void) {};
            
            /// @brief Returns the size of the compacted data
        	/// @author WVN
            virtual size_t GetPackedSize(void)=0;
            
            /// @brief Packs the data to the given buffer
            /// @param buf The buffer to pack the data to
            /// \pre Note that the size of buf must be at least the size returned by GetPackedSize()
        	/// @author WVN
            virtual void Pack(char* buf)=0;
            
            /// @brief Unpacks the data from the given buffer
            /// @param buf The buffer to unpack the data from
            /// \pre Note that the size of buf must be at least the size returned by GetPackedSize()
        	/// @author WVN
            virtual void Unpack(char* buf)=0;
            
            /// @brief Returns the rank of the sending process
        	/// @author WVN
            int Sender(void);
            
            /// @brief Returns the rank of the receiving process
        	/// @author WVN
            int Receiver(void);
            
            /// @brief Returns the priority of this transaction
        	/// @author WVN
            int Priority(void) { return priority; }
            
            /// @brief Sets the priority of this transaction
        	/// @author WVN
            void SetPriority(const int priorityValue) { priority = priorityValue; }
            
        protected:
            
            /// @brief The sending rank
            int sender;
            
            /// @brief The receiving rank
            int receiver;
            
            /// @@brief The priority of the transaction: when calling Sort() on a DataExchangePattern, transactions are
            /// sorted according to this value.
            int priority = -1;
    };
}

#endif