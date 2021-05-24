#ifndef CMF_DATA_EXCHANGE_PATTERN_H
#define CMF_DATA_EXCHANGE_PATTERN_H
#include "ParallelGroup.h"
#include "IDataTransaction.h"
#include "SingleTransaction.h"
#include "MultiTransaction.h"
#include "CartesianInterLevelBlockTransaction.h"
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
            
            /// @brief Adds a new data transaction to this data exchange pattern
            /// @param transaction A new transaction to add
            /// @param priorityLevel The priority level of the new exchange, the higher the priority, the sooner the exchange happens
            /// \pre NOTE: this exchange pattern object WILL DELETE this pointer when it is deconstructed.
        	/// @author WVN
            void Add(IDataTransaction* transaction, int priorityLevel);
            
            /// @brief Sorts the contained data transactions by their priority value
        	/// @author WVN
            void SortByPriority(void);
            
        private:
            /// @brief Computes the size of the outgoing buffer for the given rank and allocates it accordingly
            /// @param rank The rank that the outgoing data will eventually go to
        	/// @author WVN
            void ResizeOutBuffer(int rank);
            
            /// @brief Computes the size of the incoming buffer for the given rank and allocates it accordingly
            /// @param rank The rank that the incoming data will eventually come from
        	/// @author WVN
            void ResizeInBuffer(int rank);
            
            /// @brief Aggregates the information to be broadcasted using this exchange pattern and places it in a single data array per rank
        	/// @author WVN
            void Pack(void);
            
            /// @brief Deaggregates the information to received using this exchange pattern and scatters it to the appropriate arrays per rank
        	/// @author WVN
            void Unpack(void);
            
            /// @brief The group that this exchange pattern is executed over
            ParallelGroup* group;
            
            /// @brief The list of individual transactions
            /// \pre Note that this object IS RESPONSIBLE for deleting these.
            std::vector<IDataTransaction*> transactions;
            
            /// @brief Indicates whether or not the outgoing buffer reqires resizing for the corresponding rank
            std::vector<bool> resizeOutBufferRequired;
            
            /// @brief Indicates whether or not the outgoing buffer reqires resizing for the corresponding rank
            std::vector<bool> resizeInBufferRequired;
            
            /// @brief The buffer for outgoing messages
            std::vector<char*> sendBuffer;
            
            /// @brief Indicates whether or not the corresponding array in sendBuffer is allocated
            std::vector<bool> sendBufferIsAllocated;
            
            /// @brief The buffer for incoming messages
            std::vector<char*> receiveBuffer;
            
            /// @brief Indicates whether or not the corresponding array in sendBuffer is allocated
            std::vector<bool> receiveBufferIsAllocated;
            
            /// @brief Used to index outgoing and incoming buffers, preventing local allocation every time exchanges are called
            std::vector<char*> pointerIndices;
            
            /// @brief A list of buffer sizes of sendBuffer
            std::vector<size_t> sendSizes;
            
            /// @brief A list of buffer sizes of receiveBuffer
            std::vector<size_t> receiveSizes;
            
            /// @brief The list of asynchronous request handles
            std::vector<ParallelRequestHandle> requestHandles;
            
            /// @brief The list of status handles
            std::vector<ParallelStatus> statusHandles;
    };
}

#endif