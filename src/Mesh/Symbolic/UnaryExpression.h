#ifndef CMF_UNARY_EXPRESSION_H
#define CMF_UNARY_EXPRESSION_H
#include "SymbolicEvaluation.h"
namespace cmf
{
    /// @brief A class that represents a unary symbolic expression
    /// @author WVN
    class UnaryExpression : public SymbolicEvaluation
    {
        public:
            /// @brief Constructor
            /// @author WVN
            UnaryExpression(void);
            
            /// @brief Destructor
            /// @author WVN
            ~UnaryExpression(void);
            
            /// @brief Assignment operator
            /// @author WVN
            UnaryExpression& operator = (const UnaryExpression& rhs){return *this;}
        private:
    };
}

#endif