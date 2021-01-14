#include "BinaryExpression.h"
#include "UnaryExpression.h"
#include "MeshArrayExpression.h"
namespace cmf
{
    BinaryExpression::BinaryExpression(SymbolicEvaluation& left_in, BinaryOperator::BinaryOperator oper_in, SymbolicEvaluation& right_in)
    {
        oper = oper_in;
        left = left_in;
        right = right_in;
    }
    
    std::string BinaryExpression::GetExpressionString(void)
    {
        return "(" + left.GetExpressionString() + " " + BinaryOperatorStr(oper) + " " + right.GetExpressionString() + ")";
    }
    
    BinaryExpression::~BinaryExpression(void)
    {
        
    }
}