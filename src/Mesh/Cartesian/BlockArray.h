#ifndef CMF_BLOCK_ARRAY_H
#define CMF_BLOCK_ARRAY_H
#include <type_traits>
#include "ICmfHasBlockBoundIndices.h"
#include "CartesianMeshArray.h"
#include "CmfError.h"
#include "CudaStaticDefs.h"
#include "MdArray.h"
namespace cmf
{
    /// @brief A struct representing an "indexed" view of a block array. Note that the "elementRank" template parameter
    /// represents the rank only of a single element in a cell. The rank does not include the 3 spatial (i,j,k) indices.
    /// @author WVN
    template <typename arType, const int elementRank = 0> struct BlockArray : public ICmfHasBlockBoundIndices
    {
        /// @brief Gives the total rank, including the spatial indices
        static _CmfShared_ constexpr int TotalRank() {return elementRank+3;}
        /// @brief The underlying data array
        arType* data;
        /// @brief The total rank (including spatial indices)
    	const int rank = TotalRank();
        /// @brief The dimensions of the array per rank
    	int dims[TotalRank()] = {1};
        /// @brief The coefficients to multiply indices by
        int idxCoeff[TotalRank()] = {0};
        /// @brief The total index offset (accounting for exchange cells)
    	size_t idxOffset = 0;
        /// @brief Conversion from CartesianMeshArrayPointerPair
        /// @author WVN
        /// @param pointerPair The pointer pair to create this object from
        BlockArray<arType, elementRank> (const CartesianMeshArrayPointerPair& pointerPair)
        {
            data = (arType*)pointerPair.pointer;
            std::vector<int>& arDim = pointerPair.array->dims;
            int elemRankInput = pointerPair.array->rank;
            size_t elementSizeIn = SizeOfArrayType(pointerPair.array->elementType);
            if (elemRankInput!=elementRank)
            {
                CmfError("A BlockArray of rank " + std::to_string(elementRank)
                    + " is being created from array \"" + pointerPair.array->variableName
                    + "\" of rank " + std::to_string(elemRankInput));
            }
            if (sizeof(arType) != elementSizeIn)
            {
                CmfError("A Blockarray with element size " + std::to_string(sizeof(arType))
                    + " is being created from an array \"" + pointerPair.array->variableName
                    + "\" with element size " + std::to_string(elementSizeIn));
            }
            for (int i = 0; i < elemRankInput; i++)
            {
                dims[i] = arDim[i];
            }
            int* meshDataDim = &(pointerPair.array->handler->mesh->meshDataDim[0]);
            int* meshExchange = &(pointerPair.array->handler->mesh->exchangeDim[0]);
            dims[TotalRank()-3] = meshDataDim[0] + 2*meshExchange[0];
            dims[TotalRank()-2] = meshDataDim[1] + 2*meshExchange[1];
#if(CMF_IS3D)
            dims[TotalRank()-1] = meshDataDim[2] + 2*meshExchange[2];
#else
            dims[TotalRank()-1] = 1;
#endif
            idxCoeff[0] = 1;
            for (int i = 1; i < TotalRank(); i++)
            {
                idxCoeff[i] = idxCoeff[i-1]*dims[i-1];
            }
            idxOffset = meshExchange[0]*idxCoeff[TotalRank()-3];
            idxOffset += meshExchange[1]*idxCoeff[TotalRank()-2];
#if(CMF_IS3D)
            idxOffset += meshExchange[2]*idxCoeff[TotalRank()-1];
#endif
            imin = 0;
            imax = meshDataDim[0];
            exchangeI = meshExchange[0];
            jmin = 0;
            jmax = meshDataDim[1];
            exchangeJ = meshExchange[1];
            kmin = 0;
            kmax = 1;
            exchangeK = 0;
#if(CMF_IS3D)
            kmax = meshDataDim[2];
            exchangeK = meshExchange[2];
#endif
        }
        /// @brief internal indexer, base case
        /// @author WVN
        /// @param lev The current level
        /// @param t The final index
        template <typename T> _CmfShared_ inline arType * idxC(int lev, T t)
        {
            return data+idxCoeff[lev]*t;
        }

        /// @brief internal indexer, recursive case
        /// @author WVN
        /// @param lev The current level
        /// @param t The current index
        /// @param ts The remaining indices index
        template <typename T, typename... Ts> _CmfShared_ inline arType * idxC(int lev, T t, Ts... ts)
        {
            static_assert(std::is_integral<T>::value, "Integral type required for indexing");
            return idxC(lev+1, ts...) + t*idxCoeff[lev];
        }

        /// @brief indexer
        /// @author WVN
        /// @param ts The list of indices
        template <typename... Ts> _CmfShared_ inline arType & operator () (Ts... ts)
        {
            static_assert(sizeof...(Ts)==TotalRank(), "Incorrect rank in array index");
            return *(idxC(0, ts...) + idxOffset);
        }
    };
    
    /// @brief A struct representing an "indexed" view of a block array, decomposed on a byte-by-byte basis
    /// @author WVN
    struct RawUnwrappedBlockArray : public ICmfHasBlockBoundIndices
    {
        /// @brief The underlying data array
        char* data;
        /// @brief The total rank (including spatial indices)
    	const int rank = 4;
        /// @brief The dimensions of the array per rank
    	int dims[5] = {1};
        /// @brief The coefficients to multiply indices by
        int idxCoeff[5] = {0};
        /// @brief The total index offset (accounting for exchange cells)
    	size_t idxOffset = 0;
        /// @brief The total number of bytes in this array
        size_t totalSize = 0;
        
        /// @brief Conversion from CartesianMeshArrayPointerPair
        /// @author WVN
        /// @param pointerPair The pointer pair to create this object from
        RawUnwrappedBlockArray (size_t elementSize, int totalComponents, const CartesianMeshArrayPointerPair& pointerPair)
        {
            data = (char*)pointerPair.pointer;
            size_t elementSizeBytes = SizeOfArrayType(pointerPair.array->elementType);
            dims[0] = elementSize;
            dims[1] = totalComponents;
            int* meshDataDim = &(pointerPair.array->handler->mesh->meshDataDim[0]);
            int* meshExchange = &(pointerPair.array->handler->mesh->exchangeDim[0]);
            dims[2] = meshDataDim[0] + 2*meshExchange[0];
            dims[3] = meshDataDim[1] + 2*meshExchange[1];
#if(CMF_IS3D)
            dims[4] = meshDataDim[2] + 2*meshExchange[2];
#else
            dims[4] = 1;
#endif
            idxCoeff[0] = 1;
            for (int i = 1; i < 5; i++)
            {
                idxCoeff[i] = idxCoeff[i-1]*dims[i-1];
            }
            idxOffset = meshExchange[0]*idxCoeff[2];
            idxOffset += meshExchange[1]*idxCoeff[3];
#if(CMF_IS3D)
            idxOffset += meshExchange[2]*idxCoeff[4];
#endif
            imin = 0;
            imax = meshDataDim[0];
            exchangeI = meshExchange[0];
            jmin = 0;
            jmax = meshDataDim[1];
            exchangeJ = meshExchange[1];
            kmin = 0;
            kmax = 1;
            exchangeK = 0;
#if(CMF_IS3D)
            kmax = meshDataDim[2];
            exchangeK = meshExchange[2];
#endif
            totalSize = dims[0]*dims[1]*dims[2]*dims[3]*dims[4];
        }
        
        /// @brief indexer
        /// @author WVN
        /// @param ts The list of indices
        _CmfShared_ inline char & operator () (int b, int v, int i, int j, int k)
        {
            return *(data + b*idxCoeff[0] + v*idxCoeff[1] + i*idxCoeff[2] + j*idxCoeff[3] + k*idxCoeff[4] + idxOffset);
        }
    };
}

#endif