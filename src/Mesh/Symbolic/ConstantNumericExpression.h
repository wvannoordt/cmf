#ifndef CMF_CONST_NUM_EXPR_H
#define CMF_CONST_NUM_EXPR_H
#include "UnaryExpression.h"
#include "NumericType.h"
namespace cmf
{
    /// @brief A class that represents a constant numeric expression
    /// @author WVN
    class ConstantNumericExpression : public UnaryExpression
    {
        public:
            /// @brief Constructor
            /// @author WVN
            ConstantNumericExpression(void);
            
            /// @brief Destructor
            /// @author WVN
            ~ConstantNumericExpression(void);
        private:
            /// @brief The type of the numeric
            NumericType::NumericType numType;
            
            /// @brief 8 bytes of data to hold arbitrary numeric type
            char data[8];
    };
}

#endif