#include "MeshArrayExpression.h"
#include "ICmfMeshArray.h"

namespace cmf
{
    MeshArrayExpression::MeshArrayExpression(void)
    {
        arrayObject = NULL;
    }
    
    std::string MeshArrayExpression::GetExpressionString(void)
    {
        if (arrayObject) return arrayObject->GetVarName();
        else return "[?ICmfMeshArray]";
    }
    
    MeshArrayExpression& MeshArrayExpression::operator = (const MeshArrayExpression& rhs)
    {
        arrayObject = rhs.arrayObject;
    }
    
    MeshArrayExpression::MeshArrayExpression(ICmfMeshArray* arrayObject_in)
    {
        arrayObject = arrayObject_in;
    }
    
    MeshArrayExpression::~MeshArrayExpression(void)
    {
        
    }
}