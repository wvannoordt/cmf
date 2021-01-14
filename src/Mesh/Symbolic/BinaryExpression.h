#ifndef CMF_BINARY_EXPRESION_H
#define CMF_BINARY_EXPRESION_H
#include "SymbolicEvaluation.h"
#include "PTL.h"
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
    
    inline static std::string BinaryOperatorStr(int refType)
    {
        switch (refType)
        {
            case BinaryOperator::addition: return "+";
            case BinaryOperator::subtraction: return "-";
            case BinaryOperator::multiplication: return "*";
            case BinaryOperator::division: return "/";
        }
        return PTL_AUTO_ENUM_TERMINATOR;
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
            
            /// @brief Returns a string representing this symbolic evaluation
            /// @author WVN
            std::string GetExpressionString(void) override;
            
        private:
            
            /// @brief The type of binary operation being applied
            BinaryOperator::BinaryOperator oper;
            
            /// @brief The left expression
            SymbolicEvaluation left;
            
            /// @brief The right expression
            SymbolicEvaluation right;
    };
}

#endif