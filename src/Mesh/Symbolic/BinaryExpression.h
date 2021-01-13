#ifndef CMF_BINARY_EXPRESION_H
#define CMF_BINARY_EXPRESION_H
#include "SymbolicEvaluation.h"
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
    class BinaryExpression : public SymbolicEvaluation
    {
        public:
            /// @brief Constructor
            /// @param left_in The left-hand expression
            /// @param oper_in The binary operator
            /// @param right_in The right-hand expression
            /// @author WVN
            BinaryExpression(SymbolicEvaluation& left_in, BinaryOperator::BinaryOperator oper_in, SymbolicEvaluation& right_in);
            
            /// @brief Destructor
            /// @author WVN
            ~BinaryExpression(void);
        private:
            
            /// @brief The type of binary operation being applied
            BinaryOperator::BinaryOperator oper;
            
            /// @brief The left expression
            SymbolicEvaluation* left;
            
            /// @brief The right expression
            SymbolicEvaluation* right;
    };
}

#endif