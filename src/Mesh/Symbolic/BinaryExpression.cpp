#include "BinaryExpression.h"

namespace cmf
{
    BinaryExpression::BinaryExpression(SymbolicEvaluation& left_in, BinaryOperator::BinaryOperator oper_in, SymbolicEvaluation& right_in)
    {
        left = new SymbolicEvaluation();
        right = new SymbolicEvaluation();
    }
    
    BinaryExpression::~BinaryExpression(void)
    {
        delete left;
        delete right;
    }
}