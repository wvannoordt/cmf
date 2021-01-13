#ifndef ICMF_MESH_ARRAY_H
#define ICMF_MESH_ARRAY_H
#include <string>
#include "ArrayInfo.h"
#include "CmfError.h"
#include "SymbolicEvaluation.h"
#include "BinaryExpression.h"
namespace cmf
{
    class ICmfMeshArrayHandler;
    /// @brief Defines a general MeshArray object for various grid types
    /// @author WVN
    class ICmfMeshArray
    {
        public:
            /// @brief Copy constructor
            /// @param info Array information
            /// @param mesh_in The mesh over which this array is defined
            /// @author WVN
            ICmfMeshArray(ArrayInfo info);

            /// @brief Explicitly release resources used by this array
            /// @author WVN
            virtual void Destroy(void);
            
            /// @brief Returns the handler for the current array
            /// @author WVN
            virtual ICmfMeshArrayHandler* GetHandler(void);

            /// @brief Empty destructor
            /// @author WVN
            virtual ~ICmfMeshArray(void);
            
            /// @brief Returns variableName
            /// @author WVN
            std::string GetVarName(void);
            
            /// @brief Performs data exchanges to and from neighboring blocks, elements, etc.
            /// @author WVN
            virtual void Exchange(void)=0;
            
            /// @brief Addition symbolic operator
            /// @author WVN
            BinaryExpression operator + (ICmfMeshArray& rhs);
            
            /// @brief Subtraction symbolic operator
            /// @author WVN
            BinaryExpression operator - (ICmfMeshArray& rhs);
            
            /// @brief Multiplication symbolic operator
            /// @author WVN
            BinaryExpression operator * (ICmfMeshArray& rhs);
            
            /// @brief Division symbolic operator
            /// @author WVN
            BinaryExpression operator / (ICmfMeshArray& rhs);
            
            /// @brief Allows for assignment of array values based on an expression involving other arrays
            /// @author WVN
            ICmfMeshArray& operator = (const SymbolicEvaluation& rhsExpression);
            

        protected:

            /// @brief The name of the variable this array represents
            std::string variableName;

            /// @brief The rank of this array
            int rank;

            /// @brief The dimensions of this array
            int dims[MAX_RANK];
            
            /// @brief The size of a single element
            size_t elementSize;
            
            /// @brief The base pointer
            void* ptr;
    };
}

#endif
