#ifndef CMF_MD_ARRAY_H
#define CMF_MD_ARRAY_H

#include <type_traits>
#include "CmfError.h"
#include "CudaStaticDefs.h"
#include "StaticArray.h"

namespace cmf
{
    /// @brief A struct representing a generic multidimensional array
    /// @author WVN
    template <typename arType, const int arRank = 1> struct MdArray
    {
        ///@brief The base pointer
    	arType* data;
        ///@brief The array rank
    	int rank = arRank;
        ///@brief The dimensions
    	int dims[arRank];
        ///@brief The coefficients to multiply indices by
        int idxCoeff[arRank];
        ///@brief The total number of elements
        size_t totalSize;
        
        /// @brief Copy constructor
        /// @author WVN
        /// @param rhs The array assigned from
        MdArray(const MdArray& rhs)
        {
            data = rhs.data;
            rank = rhs.rank;
            totalSize = rhs.totalSize;
            for (int i = 0; i < arRank; i++)
            {
                dims[i] = rhs.dims[i];
                idxCoeff[i] = rhs.idxCoeff[i];
            }
        }
        
        /// @brief Assignment operator
        /// @param rhs The array assigned from
        /// @author WVN
        MdArray& operator =(const MdArray& rhs)
        {
            data = rhs.data;
            rank = rhs.rank;
            totalSize = rhs.totalSize;
            for (int i = 0; i < arRank; i++)
            {
                dims[i] = rhs.dims[i];
                idxCoeff[i] = rhs.idxCoeff[i];
            }
            return *this;
        }
        
        /// @brief Returns an MdArray of a different type, but the same rank. The the dimension corresponding to the provided index is reduced (or expanded) to accommodate
        /// the new type.
        /// @param index The index to change to accommodate the new type
        /// @author WVN
        template <typename arType2, const int arRank2 = 1> MdArray<arType2, arRank2> ReCast(int index)
        {
            //Expand this to reCast in rank as well?
            static_assert(arRank2==arRank, "Incorrect rank when performing ReCast");
            MdArray<arType2, arRank2> output;
            for (int i = 0; i < arRank2; i++) output.dims[i] = dims[i];
            output.dims[index] = (output.dims[index]*sizeof(arType))/sizeof(arType2);
            output.data = (arType2*)data;
            output.idxCoeff[0] = 1;
            for (int i = 1; i < arRank; i++)
            {
                output.idxCoeff[i] = output.idxCoeff[i-1]*output.dims[i-1];
            }
            return output;
        }
        
        /// @brief Constructor helper function, allows for default constructor definition
        /// @author WVN
        /// @param lev Unused parameter
    	void Ralloc(int lev)
    	{
    		totalSize = 0;
    	}
        
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
            totalSize = 1;
            for (int i = 0; i < arRank; i++)
            {
                totalSize *= dims[i];
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