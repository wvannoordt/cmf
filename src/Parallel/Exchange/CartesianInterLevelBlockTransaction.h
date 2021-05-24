#ifndef CMF_CARTESIAN_INTER_LEVEL_BLOCK_TRANSACTION_H
#define CMF_CARTESIAN_INTER_LEVEL_BLOCK_TRANSACTION_H
#include "IDataTransaction.h"
#include "Vec.h"
#include "MdArray.h"
#include "CmfPrint.h"
namespace cmf
{
    ///@brief A class representing a data transaction between two blocks of differing refinement levels
    ///@author WVN
    template <typename numType> class CartesianInterLevelBlockTransaction : public IDataTransaction
    {
        public:
            ///@brief Constructor
            ///@param sendArray_in The array that will populate the exchange region on the sending rank.
            /// \pre if the current rank is not the sending rank, it is invalid to index this array
            ///@param recvArray_in The array that will receive the exchange region on the receiving rank
            /// \pre if the current rank is not the receiving rank, it is invalid to index this array
            /// @param sendRank_in The sending rank
            /// @param recvRank_in The receiving rank
            /// @param sendBounds_in The bounding box of the exchange region, in the index-coordinate system of sendArray_in
            /// @param recvBounds_in The bounding box of the exchange region, in the index-coordinate system of recvArray_in
            /// @param exchangeSize_in The dimensions of the exchange region, in cells
            /// @param exchangeDims_in The number of exchange cells in each direction
            ///@author WVN
            CartesianInterLevelBlockTransaction
                (
                    MdArray<numType, 4> sendArray_in,
                    MdArray<numType, 4> recvArray_in,
                    int sendRank_in,
                    int recvRank_in,
                    Vec<double, 6> sendBounds_in,
                    Vec<double, 6> recvBounds_in,
                    Vec3<int> exchangeSize_in,
                    Vec3<int> exchangeDims_in
                )
                : IDataTransaction(sendRank_in, recvRank_in)
            {
                sendArray = sendArray_in;
                recvArray = recvArray_in;
                sendBounds = sendBounds_in;
                recvBounds = recvBounds_in;
                exchangeSize = exchangeSize_in;
                exchangeDims = exchangeDims_in;
                
                numComponentsPerCell = sendArray.dims[0];
            }
            
            ~CartesianInterLevelBlockTransaction(void) { }
            
            /// @brief Returns the size of the compacted data
            /// @author WVN
            virtual size_t GetPackedSize(void) override final
            {
                size_t output = 1;
                
                //The number of exchange cells
                output *= exchangeSize[0];
                output *= exchangeSize[1];
                output *= exchangeSize[2];
                
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
                int imin = (int)(recvBounds[0] - 0.5);
                int imax = (int)(recvBounds[1] + 0.5);
                int jmin = (int)(recvBounds[2] - 0.5);
                int jmax = (int)(recvBounds[3] + 0.5);
                int kmin = (int)(recvBounds[4] - 0.5);
                int kmax = (int)(recvBounds[5] + 0.5);
                int di = exchangeDims[0];
                int dj = exchangeDims[1];
                int dk = exchangeDims[2];
                size_t offset = 0;
                for (int k = kmin; k < kmax; k++)
                {
                    for (int j = jmin; j < jmax; j++)
                    {
                        for (int i = imin; i < imax; i++)
                        {
                            for (int v = 0; v < numComponentsPerCell; v++)
                            {
                                recvArray(v, i+di, j+dj, k+dk) = 15.145;
                            }
                        }
                    }
                }
            }
            
        private:
            
            /// @brief The array that will populate the exchange region on the sending rank
            MdArray<numType, 4> sendArray;
            
            /// @brief The array that will receive the exchange region on the receiving rank
            MdArray<numType, 4> recvArray;
            
            /// @brief The bounding box of the exchange region, in the index-coordinate system of sendArray_in
            Vec<double, 6> sendBounds;
            
            /// @brief The bounding box of the exchange region, in the index-coordinate system of recvArray_in
            Vec<double, 6> recvBounds;
            
            /// @brief The dimensions of the exchange region, in cells
            Vec3<int> exchangeSize;
            
            /// @brief The number of exchange cells in each direction
            Vec3<int> exchangeDims;
            
            /// @brief The number of components in a single cell
            int numComponentsPerCell;
    };
}

#endif