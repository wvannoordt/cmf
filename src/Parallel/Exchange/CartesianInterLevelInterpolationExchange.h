#ifndef CMF_CARTESIAN_INTER_LEVEL_INTERPOLATION_EXCHANGE_H
#define CMF_CARTESIAN_INTER_LEVEL_INTERPOLATION_EXCHANGE_H
#include "IDataTransaction.h"
namespace cmf
{
    /// @brief A specialized class defining an exchange pattern between two conformally-overlapping,
    /// factor-2 constrained, Cartesian blocks
	/// @author WVN
    class CartesianInterLevelInterpolationExchange : public IDataTransaction
    {
        public:
            
            /// @brief Constructor
            /// @param sendRank_in The sending rank
            /// @param recvRank_in The receiving rank
        	/// @author WVN
            CartesianInterLevelInterpolationExchange(int sendRank_in, int recvRank_in);
            
            /// @brief Returns the size of the compacted data
        	/// @author WVN
            virtual size_t GetPackedSize(void) override final;
            
            /// @brief Packs the data to the given buffer
            /// @param buf The buffer to pack the data to
            /// \pre Note that the size of buf must be at least the size returned by GetPackedSize()
        	/// @author WVN
            virtual void Pack(char* buf) override final;
            
            /// @brief Unpacks the data from the given buffer
            /// @param buf The buffer to unpack the data from
            /// \pre Note that the size of buf must be at least the size returned by GetPackedSize()
        	/// @author WVN
            virtual void Unpack(char* buf) override final;
    };
}

#endif