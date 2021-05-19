#ifndef CMF_SYMBOLIC_EVAL_H
#define CMF_SYMBOLIC_EVAL_H
#include <string>
namespace cmf
{
    /// @brief A class that represents a symbolic evaluation over a mesh
    /// @author WVN
    class SymbolicEvaluation
    {
        public:
            /// @brief Constructor
            /// @author WVN
            SymbolicEvaluation(void);
            
            /// @brief Destructor
            /// @author WVN
            ~SymbolicEvaluation(void);
            
            /// @brief Returns a string representing this symbolic evaluation
            /// @author WVN
            virtual std::string GetExpressionString(void);
            
            /// @brief Assignment operator
            /// @author WVN
            virtual SymbolicEvaluation& operator = (const SymbolicEvaluation& rhs) {return *this;}
            
        private:
            
    };
}

#endif