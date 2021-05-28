#ifndef CMF_EXACT_AMR_POSITION_H
#define CMF_EXACT_AMR_POSITION_H
#include <stdint.h>
#include "CmfPrint.h"
namespace cmf
{
    /// @brief Returns size of a byte (believe it or not, this is not constant on all systems!)
    /// @author WVN
    constexpr static int ByteSize(void) {return 8;}
    
    /// @brief Provides a means of exactly comparing the positions of blocks in an AMR tree without worrying about floating-point considerations
	/// @author WVN
    template <typename bitRepr> struct ExactAmrPosition
    {
        /// @brief 64 bits representing the position in the current partition. it bit (n-1) is 1, it is in the upper half. If bit (n-2) is 1,
        /// it is in the upper half of the upper half and so on. The most-significant bit represents the coarsest level
        bitRepr bits;
        
        /// @brief the partition that this position resides in
        int partition;
        
        /// @brief Default Constructor
        /// @author WVN
        ExactAmrPosition(void)
        {
            partition = 0;
            bits = 0;
        }
        
        /// @brief Constructor
        /// @param partition_in the partition that this position is in
        /// @param bits_in the bits to set for this position
        /// @author WVN
        ExactAmrPosition(int partition_in, bitRepr bits_in)
        {
            partition = partition_in;
            bits = bits_in;
        }
        
        /// @brief Copy Constructor
        /// @param rhs to be assigned from
        /// @author WVN
        ExactAmrPosition(const ExactAmrPosition<bitRepr> & rhs)
        {
            partition = rhs.partition;
            bits = rhs.bits;
        }
        
        /// @brief Returns the bit at the given position (note that position 0 represents the coarsest level, position (n-1) is the finest; bits are in reverse order)
        /// @param position the position of the bit to retrieve
        /// @author WVN
        char GetBit(int position) const
        {
            auto offset = GetNumBits() - 1 - position;
            bitRepr unit = 1;
            bitRepr mask = (unit<<offset);
            return (char)((bits&mask)>>offset);
        }
        
        /// @brief Returns the bit at the given position (note that position 0 represents the coarsest level, position (n-1) is the finest; bits are in reverse order)
        /// @param position the position of the bit to set
        /// @author WVN
        void SetBit(int position, int value)
        {
            auto offset = GetNumBits() - 1 - position;
            bitRepr unit = 1;
            bitRepr mask = (unit<<offset);
            bitRepr bval = value;
            bitRepr val = ((bval&1)<<offset);
            bits = (bits&~mask)|(val&mask);
        }
        
        /// @brief Returns an offset at the given AMR level
        /// @param level The level of refinement
        /// @author WVN
        bitRepr GetIntervalSize(int level)
        {
            auto offset = GetNumBits() - level;
            bitRepr unit = 1;
            return (unit<<offset);
        }
        
        /// @brief Returns the number of bits in the template parameter
        /// @author WVN
        constexpr int GetNumBits(void) const
        {
            return sizeof(bitRepr)*ByteSize();
        }
        
        /// @brief Assignment overload
        /// @param rhs to be assigned from
        /// @author WVN
        ExactAmrPosition<bitRepr>& operator =(const ExactAmrPosition<bitRepr> & rhs)
        {
            bits = rhs.bits;
            partition = rhs.partition;
            return *this;
        }
        
        /// @brief Equality operator
        /// @param rhs to compare to
        /// @author WVN
        bool operator == (const ExactAmrPosition<bitRepr> & rhs) const
        {
            return ((bits==rhs.bits)&&(partition==rhs.partition));
        }
        
        /// @brief Inquality operator
        /// @param rhs to compare to
        /// @author WVN
        bool operator < (const ExactAmrPosition<bitRepr> & rhs) const
        {
            if (partition<rhs.partition) return true;
            if (partition>rhs.partition) return false;
            return bits<rhs.bits;
        }
        
        /// @brief Inquality operator
        /// @param rhs to compare to
        /// @author WVN
        bool operator > (const ExactAmrPosition<bitRepr> & rhs) const
        {
            if (partition<rhs.partition) return false;
            if (partition>rhs.partition) return true;
            return bits>rhs.bits;
        }
        
        /// @brief Inquality operator
        /// @param rhs to compare to
        /// @author WVN
        bool operator <= (const ExactAmrPosition<bitRepr> & rhs) const
        {
            if (partition<rhs.partition) return true;
            if (partition>rhs.partition) return false;
            return bits<=rhs.bits;
        }
        
        /// @brief Inquality operator
        /// @param rhs to compare to
        /// @author WVN
        bool operator >= (const ExactAmrPosition<bitRepr> & rhs) const
        {
            if (partition<rhs.partition) return false;
            if (partition>rhs.partition) return true;
            return bits>=rhs.bits;
        }
    };
    
    ///@brief Allows streaming of ExactAmrPosition<T>
    template <typename T> static std::ostream & operator<<(std::ostream & os, const ExactAmrPosition<T> & pos)
    {
        os << "P:" << pos.partition << "[";
        T mask = 1;
        for (int i = 0; i < pos.GetNumBits(); i++)
        {
            os<<(int)(pos.GetBit(i));
        }
        os << "]";
        return os;
    }
}

#endif