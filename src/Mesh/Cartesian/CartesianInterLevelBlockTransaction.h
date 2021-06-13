#ifndef CMF_CARTESIAN_INTER_LEVEL_BLOCK_TRANSACTION_H
#define CMF_CARTESIAN_INTER_LEVEL_BLOCK_TRANSACTION_H
#include "IDataTransaction.h"
#include "Vec.h"
#include "MdArray.h"
#include "CmfPrint.h"
#include "DebugPointCloud.h"
namespace cmf
{
    namespace ExchangeOrientation
    {
        enum ExchangeOrientation
        {
            faceExchange   = 0,
            edgeExchange   = 1,
            cornerExchange = 2
        };
    }
    
    ExchangeOrientation::ExchangeOrientation ExchangeOrientationFromEdgeVector(Vec3<int>& edgeVec)
    {
        int numNonzeroEdgeComponents = 0;
        for (int i = 0; i < CMF_DIM; i++) numNonzeroEdgeComponents += ((edgeVec[i]==0)?(0):(1));
        
        int table[6] = {0, 2, 1, 0, 1, 2};
        
        return (ExchangeOrientation::ExchangeOrientation)table[(numNonzeroEdgeComponents-1)+CMF_IS3D*3];
    }
    
    ///@brief Contains properties of an inter-level mesh exchange, used to set the priority of this exchange
    ///@author WVN
    struct CartesianInterLevelExchangeProperties
    {
        
        ///@brief the orientation of this exchange
        ExchangeOrientation::ExchangeOrientation orientation = ExchangeOrientation::faceExchange;
        
        ///@brief The difference in refinement level in each direction. -1: sender is finer, 0: same level, 1: sender is coarser
        Vec3<int> levelDifference;
        
        ///@brief The edge vector from the receiver to the sender
        Vec3<int> edgeVector;
        
        ///@brief the order of interpolation
        int interpolationOrder;
        
        ///@brief Computes the priority of the relevant exchange given the properties. Note that high priority exchanges are performed first
        ///@author WVN
        int GetPriority(void)
        {
            int levelPriority = 0;
            auto priorVal = [&](int i, int j) -> int {int w[3] = {1, 2, 1}; int v[3] = {1, 2, 0}; return w[1+j]*v[i+1];};
            for (int i = 0; i < CMF_DIM; i++)
            {
                levelPriority += priorVal(levelDifference[i], edgeVector[i]);
            }
            return 12*(3-(int)orientation) + levelPriority;
        }
    };
    
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
        Vec<int, 6> indexSupport;
        
        /// @brief The origin of the box in physical space
        Vec3<double> origin;
        
        /// @brief The physical mesh spacing
        Vec3<double> dx;
        
        /// @brief Adds all points in the specified exchange region to the specified point cloud
        /// @param cloud The cloud to add the exchange region points to
        void GetExchangeRegionAsPointCloud(DebugPointCloud& cloud)
        {
            Vec3<double> dijk(0);
            for (int i = 0; i < CMF_DIM; i++) dijk[i]=((exchangeSize[i]==1)?(0.0):((bounds[2*i+1] - bounds[2*i])/(exchangeSize[i]-1)));
            for (int k = 0; k < exchangeSize[2]; k++)
            {
                for (int j = 0; j < exchangeSize[1]; j++)
                {
                    for (int i = 0; i < exchangeSize[0]; i++)
                    {
                        Vec3<double> ijk(bounds[0] + i*dijk[0], bounds[2] + j*dijk[1], bounds[4] + k*dijk[2]);
                        Vec3<double> xyz(origin[0]+ijk[0]*dx[0], origin[1]+ijk[1]*dx[1], origin[2]+ijk[2]*dx[2]);
                        cloud << xyz;
                    }
                }
            }
        }
    };
    
    ///@brief A class representing a data transaction between two blocks of differing refinement levels
    ///@author WVN
    template <typename numType> class CartesianInterLevelBlockTransaction : public IDataTransaction
    {
        public:
            ///@brief Constructor
            ///@param sendInfo_in Contains info about the sending block
            ///@param recvInfo_in Contains info about the receiving block
            ///@param exchangeProps_in Contains info about the topology of the exchange
            ///@author WVN
            CartesianInterLevelBlockTransaction
                (
                    CartesianInterLevelBlockInfo<numType>& sendInfo_in,
                    CartesianInterLevelBlockInfo<numType>& recvInfo_in,
                    CartesianInterLevelExchangeProperties& exchangeProps_in
                ) : IDataTransaction(sendInfo_in.rank, recvInfo_in.rank)
            {
                sendInfo = sendInfo_in;
                recvInfo = recvInfo_in;
                exchangeProps = exchangeProps_in;
                numComponentsPerCell = sendInfo.array.dims[0];
                
                int numDifferentRefineLevelDirections = 0;
                interpolationDirectionFor1D = 0;
                for (int i = 0; i < CMF_DIM; i++)
                {
                    numDifferentRefineLevelDirections += ((exchangeProps.levelDifference[i]==0)?(1):(0));
                    interpolationDirectionFor1D = (exchangeProps.levelDifference[i]==0)?i:interpolationDirectionFor1D;
                }
                is1DInterpolation = (numDifferentRefineLevelDirections==1);
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
            
        protected:
            
            ///@brief Contains info about the sending block
            CartesianInterLevelBlockInfo<numType> sendInfo;
            
            ///@brief Contains info about the receiving block
            CartesianInterLevelBlockInfo<numType> recvInfo;
            
            ///@brief Contains info about the details of the exchange topology
            CartesianInterLevelExchangeProperties exchangeProps;
            
            /// @brief The number of components in a single cell
            int numComponentsPerCell;
            
            /// @brief Indicates if this operator is a 1-Dimensional interpolation operator
            bool is1DInterpolation;
            
            /// @brief If this exchange is a 1-D interpolation, this is the direction of interpolation
            int interpolationDirectionFor1D;
    };
}

#endif