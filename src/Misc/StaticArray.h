#ifndef CMF_STATIC_ARRAY_H
#define CMF_STATIC_ARRAY_H
#include <type_traits>
namespace cmf
{
    ///@brief Represents a zero-overhead generic static multi-dimensional array
    ///@author WVN
    template <typename arType, const int... dims> class StaticArray
    {
    	private:
            
            ///@brief Static product base case
            ///@author WVN
    		template <const int depth, typename ...T> static constexpr size_t sprodr(void) {return 1;}
            ///@brief Recursive static product base case
            ///@author WVN
    		template <const int depth, const int T, const int... Tss> static constexpr size_t sprodr(void) {return (depth==0)?T:T*sprodr<depth-1, Tss...>();}
            ///@brief Computes the product of template parameters
            ///@author WVN
    		template <const int depth, const int... Tss> static constexpr size_t sprod(void) {return (depth==0)?1:sprodr<depth-1, Tss...>();}
            ///@brief Underlying array
    		arType data[StaticArray::sprod<sizeof...(dims), dims...>()];
            ///@brief Base indexer
            ///@author WVN
    		template <const int depth, typename index> inline size_t idxR(index i)
    		{
    			static_assert(std::is_integral<index>::value, "Integral value for index required.");
    			return i*sprod<sizeof...(dims)-1, dims...>();
    		}
            ///@brief Recursive indexer
            ///@author WVN
    		template <const int depth, typename index, typename... indices> inline size_t idxR(index i, indices... is)
    		{
    			static_assert(std::is_integral<index>::value, "Integral value for index required.");
    			return sprod<depth, dims...>()*i + idxR<depth+1>(is...);
    		}
    	public:
            ///@brief Index operator
            ///param index parameter pack
            ///@author WVN
    		template <typename... indices> inline arType& operator() (indices... is) {return data[idxR<0>(is...)];}
            ///@brief Constructor
            ///@author WVN
            StaticArray(void) {}
            
            ///@brief Initialization constructor
            ///@param val Value to fill with
            ///@author WVN
            StaticArray(const arType val)
            {
                for (size_t i = 0; i < StaticArray::sprod<sizeof...(dims), dims...>(); i++)
                {
                    data[i] = val;
                }
            }
            
            ///@brief Assignment operator
            ///@param val Value to fill with
            ///@author WVN
            StaticArray& operator = (const arType val)
            {
                for (size_t i = 0; i < StaticArray::sprod<sizeof...(dims), dims...>(); i++)
                {
                    data[i] = val;
                }
                return *this;
            }
            
    };
}
#endif