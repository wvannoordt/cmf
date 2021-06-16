#ifndef CMF_CARTESIAN_INTER_LEVEL_FACE_TRANSACTION_H
#define CMF_CARTESIAN_INTER_LEVEL_FACE_TRANSACTION_H

#include "CartesianInterLevelBlockTransaction.h"

namespace cmf
{
    
    ///@brief A class representing a data transaction between two blocks of differing refinement levels, oriented on a face
    ///@author WVN
    template <typename numType> class CartesianInterLevelFaceTransaction : public CartesianInterLevelBlockTransaction<numType>
    {
        
        using CartesianInterLevelBlockTransaction<numType>::sendInfo;
        using CartesianInterLevelBlockTransaction<numType>::recvInfo;
        using CartesianInterLevelBlockTransaction<numType>::exchangeProps;
        using CartesianInterLevelBlockTransaction<numType>::numComponentsPerCell;
        using CartesianInterLevelBlockTransaction<numType>::is1DInterpolation;
        using CartesianInterLevelBlockTransaction<numType>::interpolationDirectionFor1D;
        
        public:
            ///@brief Constructor
            ///@param sendInfo_in Contains info about the sending block
            ///@param recvInfo_in Contains info about the receiving block
            ///@param exchangeProps_in Contains info about the topology of the exchange
            ///@author WVN
            CartesianInterLevelFaceTransaction
                (
                    CartesianInterLevelBlockInfo<numType>& sendInfo_in,
                    CartesianInterLevelBlockInfo<numType>& recvInfo_in,
                    CartesianInterLevelExchangeProperties& exchangeProps_in
                ) : CartesianInterLevelBlockTransaction<numType>(sendInfo_in, recvInfo_in, exchangeProps_in)
            {
                is1DinNormalDirection = is1DInterpolation&&(exchangeProps_in.edgeVector[interpolationDirectionFor1D]==0);
            }
            
            /// @brief Packs the data to the given buffer
            /// @param buf The buffer to pack the data to
            /// \pre Note that the size of buf must be at least the size returned by GetPackedSize()
            /// @author WVN
            virtual void Pack(char* buf) override final
            {
                if (is1DinNormalDirection)
                {
                    PackAs1DInNormalDirection(buf);
                }
                else if (is1DInterpolation)
                {
                    PackAs1DInTangentialDirection(buf);
                }
                else
                {
                    PackAsMultiDimensional(buf);
                }
            }
            
            ///@brief Unpacks the data assuming a 1-D normal interpolation operator
            ///@param buf the buffer to unpack from
            ///@author WVN
            void UnpackAs1DInNormalDirection(char* buf)
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
                                recvInfo.array(v, i+di, j+dj, k+dk) = 1.0;
                            }
                        }
                    }
                }
            }
            
            ///@brief Packs the data assuming a 1-D normal interpolation operator
            ///@param buf the buffer to pack to
            ///@author WVN
            void PackAs1DInNormalDirection(char* buf)
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
                                recvInfo.array(v, i+di, j+dj, k+dk) = 1.0;
                            }
                        }
                    }
                }
            }
            
            ///@brief Unpacks the data assuming a 1-D tangential interpolation operator
            ///@param buf the buffer to unpack from
            ///@author WVN
            void UnpackAs1DInTangentialDirection(char* buf)
            {
                //Sender is current rank
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
                                recvInfo.array(v, i+di, j+dj, k+dk) = 2.0;
                            }
                        }
                    }
                }
            }
            
            ///@brief Packs the data assuming a 1-D tangential interpolation operator
            ///@param buf the buffer to pack to
            ///@author WVN
            void PackAs1DInTangentialDirection(char* buf)
            {
                
            }
            
            ///@brief Unpacks the data assuming a multidimensional interpolation operator
            ///@param buf the buffer to unpack from
            ///@author WVN
            void UnpackAsMultiDimensional(char* buf)
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
                                recvInfo.array(v, i+di, j+dj, k+dk) = 3.0;
                            }
                        }
                    }
                }
            }
            
            ///@brief Packs the data assuming a multidimensional interpolation operator
            ///@param buf the buffer to pack to
            ///@author WVN
            void PackAsMultiDimensional(char* buf)
            {
                
            }
            
            
            /// @brief Unpacks the data from the given buffer
            /// @param buf The buffer to unpack the data from
            /// \pre Note that the size of buf must be at least the size returned by GetPackedSize()
            /// @author WVN
            virtual void Unpack(char* buf) override final
            {
                if (is1DinNormalDirection)
                {
                    UnpackAs1DInNormalDirection(buf);
                }
                else if (is1DInterpolation)
                {
                    UnpackAs1DInTangentialDirection(buf);
                }
                else
                {
                    UnpackAsMultiDimensional(buf);
                }
            }
            
        private:
            
            ///@brief indicates whether or not this exchange is 1-dimensional in the tangential direction
            bool is1DinNormalDirection;

    };
}

#endif