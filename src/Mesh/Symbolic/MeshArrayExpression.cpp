#include "MeshArrayExpression.h"
#include "ICmfMeshArray.h"

namespace cmf
{
    MeshArrayExpression::MeshArrayExpression(void)
    {
        
    }
    
    MeshArrayExpression::MeshArrayExpression(ICmfMeshArray* arrayObject_in)
    {
        arrayObject = arrayObject_in;
    }
    
    MeshArrayExpression::~MeshArrayExpression(void)
    {
        
    }
}