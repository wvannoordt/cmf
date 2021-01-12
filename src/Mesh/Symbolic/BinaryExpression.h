#ifndef CMF_BINARY_EXPRESION_H
#define CMF_BINARY_EXPRESION_H

namespace cmf
{
    namespace BinaryOperator
    {
        enum BinaryOperator
        {
            addition,
            subtraction,
            multiplication,
            division
        };
    }
    
    /// @brief A class that represents a binary symbolic expression
    /// @author WVN
    class BinaryExpression
    {
        public:
            /// @brief Constructor
            /// @author WVN
            BinaryExpression(void);
            
            /// @brief Destructor
            /// @author WVN
            ~BinaryExpression(void);
        private:
            
            /// @brief The type of binary operation being applied
            BinaryOperator::BinaryOperator oper;
    };
}

#endif