#ifndef CMF_BLOCK_ARRAY_H
#define CMF_BLOCK_ARRAY_H
#include <type_traits>
#include "ICmfHasBlockBoundIndices.h"
#include "CartesianMeshArray.h"
#include "CmfError.h"
#include "CudaStaticDefs.h"
namespace cmf
{
    /// @brief A struct representing an "indexed" view of a block array. Note that the "elementRank" template parameter
    /// represents the rank only of a single element in a cell. The rank does not include the 3 spatial (i,j,k) indices.
    /// @author WVN
    template <typename arType, const int elementRank = 0> struct BlockArray : public ICmfHasBlockBoundIndices
    {
        /// @brief Gives the total rank, including the spatial indices
        static constexpr int TotalRank() {return elementRank+3;}
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
            int* meshDataDim = pointerPair.array->handler->mesh->meshDataDim;
            int* meshExchange = pointerPair.array->handler->mesh->exchangeDim;
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
    
    /// @brief A struct representing a generic multidimensional array
    /// @author WVN
    template <typename arType, const int arRank = 1> struct MdArray
    {
    	arType* data;
    	int rank = arRank;
    	int dims[arRank];
        int idxCoeff[arRank];
        
        /// @brief Constructor function
        /// @author WVN
        /// @param lev The recursive level
        /// @param t the final index dimension
    	template <typename Ts> void Ralloc(int lev, Ts t)
    	{
    		dims[lev] = t;
            idxCoeff[0] = 1;
            for (int i = 1; i < arRank; i++)
            {
                idxCoeff[i] = idxCoeff[i-1]*dims[i-1];
            }
    	}
        
        /// @brief Constructor function
        /// @author WVN
        /// @param lev The recursive level
        /// @param t the final index dimension
        /// @param ts the remaining indices
    	template <typename T, typename... Ts> void Ralloc(int lev, T t, Ts... ts)
    	{
    		static_assert(std::is_integral<T>::value, "Integral type required for dimension initialization.");
    		dims[lev] = t;
    		Ralloc(lev+1, ts...);
    	}
        
        /// @brief Constructor
        /// @author WVN
        /// @param ts the remaining indices
    	template <typename... Ts> MdArray(Ts... ts)
    	{
    		Ralloc(0, ts...);
    	}
        
        /// @brief Constructor, including external data pointer
        /// @author WVN
        /// @param ptr The pointer to the data
        /// @param ts the remaining indices
    	template <typename... Ts> MdArray(arType* ptr, Ts... ts)
    	{
            data = ptr;
    		Ralloc(0, ts...);
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
            static_assert(sizeof...(Ts)==arRank, "Incorrect rank in array index");
            return *(idxC(0, ts...));
        }
        
        /// @brief internal offset calculation, base case
        /// @author WVN
        /// @param lev The current level
        /// @param t The final index
        template <typename T> _CmfShared_ inline size_t offsetInternal(int lev, T t)
        {
            return idxCoeff[lev]*t;
        }

        /// @brief internal offset calculation, recursive case
        /// @author WVN
        /// @param lev The current level
        /// @param t The current index
        /// @param ts The remaining indices index
        template <typename T, typename... Ts> _CmfShared_ inline size_t offsetInternal(int lev, T t, Ts... ts)
        {
            static_assert(std::is_integral<T>::value, "Integral type required for indexing");
            return offsetInternal(lev+1, ts...) + t*idxCoeff[lev];
        }

        /// @brief Returns the offset of an element
        /// @author WVN
        /// @param ts The list of indices
        template <typename... Ts> _CmfShared_ inline size_t offset(Ts... ts)
        {
            static_assert(sizeof...(Ts)==arRank, "Incorrect rank in array index");
            return offsetInternal(0, ts...);
        }
    };
}

#endif