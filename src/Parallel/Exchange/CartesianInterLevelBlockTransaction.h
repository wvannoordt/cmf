#ifndef CMF_CARTESIAN_INTER_LEVEL_BLOCK_TRANSACTION_H
#define CMF_CARTESIAN_INTER_LEVEL_BLOCK_TRANSACTION_H
#include "IDataTransaction.h"
#include "Vec.h"
#include "MdArray.h"
#include "CmfPrint.h"
namespace cmf
{
    ///@brief Contains constructor info for CartesianInterLevelBlockTransaction, to avoid too many constructor arguments
    ///@author WVN
    template <typename numType> struct CartesianInterLevelBlockInfo
    {
        ///@brief The corresponding block array
        /// \pre if the current rank is not "rank", it is invalid to index this array
        MdArray<numType, 4> array;
        
        ///@brief The rank that owns "array"
        int rank;
        
        ///@brief The index bounding box of the exchange region
        Vec<double, 6> bounds;
        
        ///@brief The size of the exchange region in cells
        Vec3<int> exchangeSize;
        
        ///@brief The dimensions of the exchange cells on the block (not necessarily he same as exchangeSize)
        Vec3<int> exchangeDims;
        
        
        // (2-D example)
        //                |<-- numSupportingExchangeCells(2*i) -->|<--       exchange region, direction i        -->|<--numSupportingExchangeCells(2*i+1)-->|
        //
        //                                                        |                                                 |
        //                                                        |                                                 |
        //                +---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        //                |         |         |         |         |         |         |         |         |         |         |         |         |         |
        //                |    +    |    +    |    +    |    +    |    +    |    +    |    +    |    +    |    +    |    +    |    +    |    +    |    +    |
        //                |         |         |         |         |         |         |         |         |         |         |         |         |         |
        //                +---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        //                |         |         |         |         |         |         |         |         |         |         |         |         |         |
        //                |    +    |    +    |    +    |    +    |    +    |    +    |    +    |    +    |    +    |    +    |    +    |    +    |    +    |
        //                |         |         |         |         |         |         |         |         |         |         |         |         |         |
        //                +---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        //                                                        |                                                 |
        //                                                        |                                                 |
        
        /// @brief Contains the number of cells available for interpolation beyond the corresponding index bound in "bounds" (see relevant sketch in source)
        Vec<int, 6> numSupportingExchangeCells;
        
        /// @brief The physical bounding box of the relevant block (for debugging)
        Vec<double, 6> spatialBounds;
    };
    
    ///@brief A class representing a data transaction between two blocks of differing refinement levels
    ///@author WVN
    template <typename numType> class CartesianInterLevelBlockTransaction : public IDataTransaction
    {
        public:
            ///@brief Constructor
            ///@param sendInfo_in Contains info about the sending block
            ///@param recvInfo_in Contains info about the receiving block
            ///@author WVN
            CartesianInterLevelBlockTransaction(CartesianInterLevelBlockInfo<numType>& sendInfo_in, CartesianInterLevelBlockInfo<numType>& recvInfo_in) : IDataTransaction(sendInfo_in.rank, recvInfo_in.rank)
            {
                sendInfo = sendInfo_in;
                recvInfo = recvInfo_in;
                numComponentsPerCell = sendInfo.array.dims[0];
            }
            
            ~CartesianInterLevelBlockTransaction(void) { }
            
            /// @brief Returns the size of the compacted data
            /// @author WVN
            virtual size_t GetPackedSize(void) override final
            {
                size_t output = 1;
                
                //The number of exchange cells
                output *= sendInfo.exchangeSize[0];
                output *= sendInfo.exchangeSize[1];
                output *= sendInfo.exchangeSize[2];
                
                //The size of a single cell
                output*=sizeof(numType);
                output*=numComponentsPerCell;
                
                return output;
            }
            
            /// @brief Packs the data to the given buffer
            /// @param buf The buffer to pack the data to
            /// \pre Note that the size of buf must be at least the size returned by GetPackedSize()
            /// @author WVN
            virtual void Pack(char* buf) override final
            {
                
            }
            
            /// @brief Unpacks the data from the given buffer
            /// @param buf The buffer to unpack the data from
            /// \pre Note that the size of buf must be at least the size returned by GetPackedSize()
            /// @author WVN
            virtual void Unpack(char* buf) override final
            {
                int imin = (int)(recvInfo.bounds[0] - 0.5);
                int imax = (int)(recvInfo.bounds[1] + 0.5);
                int jmin = (int)(recvInfo.bounds[2] - 0.5);
                int jmax = (int)(recvInfo.bounds[3] + 0.5);
                int kmin = (int)(recvInfo.bounds[4] - 0.5);
                int kmax = (int)(recvInfo.bounds[5] + 0.5);
                int di = recvInfo.exchangeDims[0];
                int dj = recvInfo.exchangeDims[1];
                int dk = recvInfo.exchangeDims[2];
                size_t offset = 0;
                for (int k = kmin; k < kmax; k++)
                {
                    for (int j = jmin; j < jmax; j++)
                    {
                        for (int i = imin; i < imax; i++)
                        {
                            for (int v = 0; v < numComponentsPerCell; v++)
                            {
                                recvInfo.array(v, i+di, j+dj, k+dk) = 15.145;
                            }
                        }
                    }
                }
            }
            
            ///@brief Returns a struct containing information about the sending block
            ///@author WVN
            CartesianInterLevelBlockInfo<numType> GetSendInfo()
            {
                return sendInfo;
            }
            
            ///@brief Returns a struct containing information about the receiving block
            ///@author WVN
            CartesianInterLevelBlockInfo<numType> GetRecvInfo()
            {
                return recvInfo;
            }
            
        private:
            
            ///@brief Contains info about the sending block
            CartesianInterLevelBlockInfo<numType> sendInfo;
            
            ///@brief Contains info about the receiving block
            CartesianInterLevelBlockInfo<numType> recvInfo;
            
            /// @brief The number of components in a single cell
            int numComponentsPerCell;
    };
}

#endif