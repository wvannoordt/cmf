#ifndef CMF_DATA_EXCHANGE_PATTERN_H
#define CMF_DATA_EXCHANGE_PATTERN_H
#include "ParallelGroup.h"
#include "IDataTransaction.h"
#include "SingleTransaction.h"
#include <vector>
#include "CmfGC.h"
namespace cmf
{
    /// @brief Defines a series of data transactions that can be repeated, and also a good reference if you want to 
    /// know how to spell "receive"
	/// @author WVN
    class DataExchangePattern
    {
        public:
            /// @brief Constructor
            /// @param group_in the group that this exchange pattern is executed over
        	/// @author WVN
            DataExchangePattern(ParallelGroup* group_in);
            
            /// @brief Destructor
        	/// @author WVN
            ~DataExchangePattern(void);
            
            /// @brief Performs data exchanges
        	/// @author WVN
            void ExchangeData(void);
            
            /// @brief Adds a new data transaction to this data exchange pattern
            /// @param transaction A new transaction to add
            /// \pre NOTE: this exchange pattern object WILL DELETE this pointer when it is deconstructed.
        	/// @author WVN
            void Add(IDataTransaction* transaction);
            
        private:
            /// @brief Computes the size of the outgoing buffer and allocates it accordingly
        	/// @author WVN
            void ResizeOutBuffer(void);
            
            /// @brief Computes the size of the incoming buffer and allocates it accordingly
        	/// @author WVN
            void ResizeInBuffer(void);
            
            /// @brief Aggregates the information to be broadcasted using this exchange pattern and places it in a single data array
        	/// @author WVN
            void Pack(void);
            
            /// @brief Deaggregates the information to received using this exchange pattern and scatters it to the appropriate arrays
        	/// @author WVN
            void Unpack(void);
            
            /// @brief The group that this exchange pattern is executed over
            ParallelGroup* group;
            
            /// @brief The list of individual transactions
            /// \pre Note that this object IS RESPONSIBLE for deleting these.
            std::vector<IDataTransaction*> transactions;
            
            /// @brief Indicates whether or not the outgoing buffer reqires resizing
            bool resizeOutBufferRequired;
            
            /// @brief Indicates whether or not the outgoing buffer reqires resizing
            bool resizeInBufferRequired;
            
            /// @brief The buffer for outgoing messages
            char* sendBuffer;
            
            /// @brief Indicates whether or not sendBuffer is allocated
            bool sendBufferIsAllocated;
            
            /// @brief The buffer for incoming messages
            char* receiveBuffer;
            
            /// @brief Indicates whether or not sendBuffer is allocated
            bool receiveBufferIsAllocated;
    };
}

#endif