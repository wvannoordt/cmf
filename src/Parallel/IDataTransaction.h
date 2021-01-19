#ifndef CMF_IDATA_TRANSACTION_H
#define CMF_IDATA_TRANSACTION_H
#include "CmfGC.h"
namespace cmf
{
    /// @brief Defines a generic parallel data transaction (including self-to-self transactions)
	/// @author WVN
    class IDataTransaction
    {
        public:
            /// @brief Constructor
        	/// @author WVN
            IDataTransaction(void);
            
            /// @brief Destructor
        	/// @author WVN
            ~IDataTransaction(void);
            
            /// @brief Returns the size of the compacted data
        	/// @author WVN
            virtual size_t GetPackedSize(void)=0;
            
            /// @brief Packs the data to the given buffer
            /// @param buf The buffer to pack the data to
            /// \pre Note that the size of buf must be at least the size returned by GetPackedSize()
        	/// @author WVN
            virtual void Pack(char* buf)=0;
            
            /// @brief Returns the rank of the sending process
        	/// @author WVN
            virtual int Sender(void)=0;
            
            /// @brief Returns the rank of the receiving process
        	/// @author WVN
            virtual int Receiver(void)=0;
            
        private:
    };
}

#endif