#ifndef CMF_DATA_EXCHANGE_PATTERN_H
#define CMF_DATA_EXCHANGE_PATTERN_H
#include "ParallelGroup.h"
namespace cmf
{
    /// @brief Defines a series of data transactions that can be repeated
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
        private:
            
            /// @brief Aggregates the information to be broadcasted using this exchange pattern and places it in a single data array
        	/// @author WVN
            void Pack(void);
            
            /// @brief Deaggregates the information to received using this exchange pattern and scatters it to the appropriate arrays
        	/// @author WVN
            void Unpack(void);
            
            /// @brief the group that this exchange pattern is executed over
            ParallelGroup* group;
    };
}

#endif