#pragma once
#include "CartesianMeshArray.h"
#include "MdArray.h"
namespace cmf
{
    namespace detail::cartesian
    {
        template <typename ltype, typename rtype, class element_operand_t>
        static void inplace_elementwise_binary_op(CartesianMeshArray& a1, CartesianMeshArray& a2, const element_operand_t& func)
        {
            for (auto lb: a1)
            {
                ltype* p1 = (ltype*)a1.GetNodePointerWithNullDefault(lb);
                rtype* p2 = (rtype*)a2.GetNodePointerWithNullDefault(lb);
                std::size_t elemCount = a1.GetArraySizePerBlock();
                for (std::size_t idx = 0; idx < elemCount; idx++)
                {
                    func(*(p1+idx), *(p2+idx));
                }
            }
        }
        template <typename ltype, class element_operand_t>
        static void inplace_elementwise_unary_op(CartesianMeshArray& a1, const element_operand_t& func)
        {
            for (auto lb: a1)
            {
                ltype* p1 = (ltype*)a1.GetNodePointerWithNullDefault(lb);
                std::size_t elemCount = a1.GetArraySizePerBlock();
                for (std::size_t idx = 0; idx < elemCount; idx++)
                {
                    func(*(p1+idx));
                }
            }
        }
    }
}