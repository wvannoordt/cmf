#ifndef CMF_CARTESIAN_INTER_LEVEL_BLOCK_TRANSACTION_H
#define CMF_CARTESIAN_INTER_LEVEL_BLOCK_TRANSACTION_H
#include "IDataTransaction.h"
#include "Vec.h"
#include "MdArray.h"
#include "CmfPrint.h"
#include "DebugPointCloud.h"
#include "InterpolationOperator1D.h"
#include "RefinementTreeNode.h"
#include "ComputeDevice.h"
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
    
    static ExchangeOrientation::ExchangeOrientation ExchangeOrientationFromEdgeVector(Vec3<int>& edgeVec)
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
        ComputeDevice rank;
        
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
        
        /// @brief The node associated with this exchange
        RefinementTreeNode* node;
        
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
                CreateInterpolationOperators();
            }
            
            ~CartesianInterLevelBlockTransaction(void) { }
            
            /// @brief Creates the interpolation operators, e.g. stores the coordinates of the interpolation supports
            /// @author WVN
            void CreateInterpolationOperators(void)
            {
                for (int i = 0; i < CMF_DIM; i++)
                {
                    CreateSingleInterpolationOperator(sendOperators[i], sendInfo, exchangeProps, i);
                }
            }
            
            /// @brief Creates a single, 1D interpolation operator provided information about the topology of an exchange
            /// @param interp The interpolation operator to create (output)
            /// @param info contains information about the block structure of the relevant array
            /// @param props contains information about the properties of this exchange \see CartesianInterLevelExchangeProperties
            /// @param idir The cartesian direction relative to info.array
            /// @author WVN
            void CreateSingleInterpolationOperator(InterpolationOperator1D& interp, CartesianInterLevelBlockInfo<numType>& info, CartesianInterLevelExchangeProperties& props, int idir)
            {
                double imin = info.bounds[2*idir]     - 0.5;
                double imax = info.bounds[2*idir + 1] + 0.5;
                int iLowerBound = 0;
                int iUpperBound = info.array.dims[idir+1] - 2*info.exchangeDims[idir];
                
                //more adjustment required here.
                
                //ugly
                
                interp.SetSize(iUpperBound-iLowerBound+1);
                interp.FillData(0.0);
                for (int i = 0; i < iUpperBound-iLowerBound+1; i++)
                {
                    interp.coords[i] = (double)(iLowerBound+i)+info.exchangeDims[idir];
                }
                interp.order = props.interpolationOrder;
            }
            
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
            
            /// @brief Interpolates the data on the provided block at the given coordinates using the provided operators
            /// @param ijk Coordinates to interpolate to. Note that cell centers assume integer values, while faces are half-integers
            /// @param operators An array of operator objects used for the multi-dimensional interpolation
            /// @param array The array to interpolate. Note that this array should include exchange cells
            /// @param exchangeDims the dimensions of the exchange cells of array
            /// @param var The cell variable to interpolate
            /// @author WVN
            inline numType InterpolateAt(Vec3<double>& ijk, InterpolationOperator1D (&operators)[CMF_DIM], Vec3<int> exchangeDims, MdArray<numType, 4>& array, int var)
            {
                Vec3<int> startIdx;
                for (int d = 0; d < CMF_DIM; d++)
                {
                    startIdx[d] = operators[d].FindMinStencilIndex(ijk[d]);
                }
                numType output = 0.0;
                Vec3<double> coeff = 1.0;
                Vec3<int> idx;
                for (int k = 0; k < (CMF_IS3D?operators[CMF_DIM-1].order:1); k++)
                {
                    idx[2] = k;
                    for (int j = 0; j < operators[1].order; j++)
                    {
                        idx[1] = j;
                        for (int i = 0; i < operators[0].order; i++)
                        {
                            idx[0] = i;
                            for (int d = 0; d < CMF_DIM; d++)
                            {
                                coeff[d] = operators[d].GetCoefficientAtPoint(startIdx[d], idx[d], ijk[d]);
                            }
                            Vec3<double> dataIdx = 0;
                            Vec3<int> dataIndex = 0;
                            bool isValidData = true;
                            auto abs = [](double z) ->double {return (z<0)?(-z):z;};
                            for (int d = 0; d < CMF_DIM; d++)
                            {
                                dataIdx[d] = operators[d].coords[startIdx[d]+idx[d]];
                                int didx = floor(dataIdx[d]);
                                dataIndex[d] = didx;
                                isValidData = isValidData && abs(dataIdx[d]-didx)<1e-3 && didx>=exchangeDims[d] && didx<exchangeDims[d]+array.dims[1+d];
                            }
                            double data = 0.0;
                            if (isValidData) data = array(var, dataIndex[0], dataIndex[1], dataIndex[2]);
                            output += coeff[0]*coeff[1]*coeff[2]*data;
                        }
                    }
                }
                return output;
            }
            
            /// @brief Packs the data to the given buffer
            /// @param buf The buffer to pack the data to
            /// \pre Note that the size of buf must be at least the size returned by GetPackedSize()
            /// @author WVN
            virtual void Pack(char* buf) override final
            {
                numType* numTypeBuf = (numType*)buf;
                size_t offset = 0;
                Vec3<double> dijk = 0;
                Vec3<double> ijk = 0;
                for (int d = 0; d < CMF_DIM; d++)
                {
                    dijk[d] = (sendInfo.exchangeSize[d]==1)?(0.0):((sendInfo.bounds[2*d+1] - sendInfo.bounds[2*d])/(sendInfo.exchangeSize[d]-1));
                }
                for (int k = 0; k < sendInfo.exchangeSize[2]; k++)
                {
                    ijk[2] = sendInfo.bounds[4]+k*dijk[2] - 0.5 + sendInfo.exchangeDims[2];
                    for (int j = 0; j < sendInfo.exchangeSize[1]; j++)
                    {
                        ijk[1] = sendInfo.bounds[2]+j*dijk[1] - 0.5 + sendInfo.exchangeDims[1];
                        for (int i = 0; i < sendInfo.exchangeSize[0]; i++)
                        {
                            ijk[0] = sendInfo.bounds[0]+i*dijk[0] - 0.5 + sendInfo.exchangeDims[0];
                            for (int v = 0; v < numComponentsPerCell; v++)
                            {
                                numTypeBuf[offset++] = InterpolateAt(ijk, sendOperators, sendInfo.exchangeDims, sendInfo.array, v);
                            }
                        }
                    }
                }
            }
            
            /// @brief Unpacks the data from the given buffer
            /// @param buf The buffer to unpack the data from
            /// \pre Note that the size of buf must be at least the size returned by GetPackedSize()
            /// @author WVN
            virtual void Unpack(char* buf) override final
            {
                numType* numTypeBuf = (numType*)buf;
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
                                Vec3<double> ijk(i+di, j+dj, k+dk);
                                recvInfo.array(v, i+di, j+dj, k+dk) = InterpolateAt(ijk, recvOperators, recvInfo.exchangeDims, recvInfo.array, v) + numTypeBuf[offset++];
                            }
                        }
                    }
                }
            }
            
        protected:
            
            ///@brief Contains info about the sending block
            CartesianInterLevelBlockInfo<numType> sendInfo;
            
            ///@brief Contains info about the receiving block
            CartesianInterLevelBlockInfo<numType> recvInfo;
            
            ///@brief Contains info about the details of the exchange topology
            CartesianInterLevelExchangeProperties exchangeProps;
            
            ///@brief An array of 1-D interpolation operators used to construct exchange cell values on the sending block
            InterpolationOperator1D sendOperators[CMF_DIM];
            
            ///@brief An array of 1-D interpolation operators used to construct exchange cell values on the receiving block
            InterpolationOperator1D recvOperators[CMF_DIM];
            
            /// @brief The number of components in a single cell
            int numComponentsPerCell;
            
            /// @brief Indicates if this operator is a 1-Dimensional interpolation operator
            bool is1DInterpolation;
            
            /// @brief If this exchange is a 1-D interpolation, this is the direction of interpolation
            int interpolationDirectionFor1D;
    };
}

#endif