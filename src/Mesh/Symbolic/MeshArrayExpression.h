#ifndef CMF_MESH_ARRAY_EXPRESSION_H
#define CMF_MESH_ARRAY_EXPRESSION_H
#include "UnaryExpression.h"
namespace cmf
{
    class ICmfMeshArray;
    /// @brief A class that represents an expression pointing to a mesh array
    /// @author WVN
    class MeshArrayExpression : public UnaryExpression
    {
        public:
            /// @brief Constructor
            /// @author WVN
            MeshArrayExpression(void);
            
            /// @brief Constructor
            /// @param arrayObject_in The mesh array to point to
            /// @author WVN
            MeshArrayExpression(ICmfMeshArray* arrayObject_in);
            
            /// @brief Destructor
            /// @author WVN
            ~MeshArrayExpression(void);
        private:
            
            /// @brief The mesh array that this expression points to
            ICmfMeshArray* arrayObject;        
    };
}

#endif