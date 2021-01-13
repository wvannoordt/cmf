#include "ICmfMeshArray.h"
#include "CmfScreen.h"
#include "MeshArrayExpression.h"
#include "BinaryExpression.h"
namespace cmf
{
    ICmfMeshArray::ICmfMeshArray(ArrayInfo info)
    {
        variableName = info.name;
        if (info.rank > MAX_RANK) CmfError("Rank of variable \"" + variableName + "\" exceeds MAX_RANK (" + std::to_string(MAX_RANK) + "): recompile with greater limit.");
        for (int i = 0; i < info.rank; i++)
        {
            dims[i] = info.dimensions[i];
        }
        rank = info.rank;
        elementSize = info.elementSize;
    }
    
    ICmfMeshArrayHandler* ICmfMeshArray::GetHandler(void)
    {
        return NULL;
    }
    
    std::string ICmfMeshArray::GetVarName(void)
    {
        return variableName;
    }
    
    void ICmfMeshArray::Destroy(void)
    {
        
    }
    
    ICmfMeshArray& ICmfMeshArray::operator = (const SymbolicEvaluation& rhsExpression)
    {
        return *this;
    }
    
    BinaryExpression ICmfMeshArray::operator + (ICmfMeshArray& rhs)
    {
        MeshArrayExpression lval(this);
        MeshArrayExpression rval(&rhs);
        return BinaryExpression(lval, BinaryOperator::addition, rval);
    }
    
    BinaryExpression ICmfMeshArray::operator - (ICmfMeshArray& rhs)
    {
        MeshArrayExpression lval(this);
        MeshArrayExpression rval(&rhs);
        return BinaryExpression(lval, BinaryOperator::subtraction, rval);
    }
    
    BinaryExpression ICmfMeshArray::operator * (ICmfMeshArray& rhs)
    {
        MeshArrayExpression lval(this);
        MeshArrayExpression rval(&rhs);
        return BinaryExpression(lval, BinaryOperator::multiplication, rval);
    }
    
    BinaryExpression ICmfMeshArray::operator / (ICmfMeshArray& rhs)
    {
        MeshArrayExpression lval(this);
        MeshArrayExpression rval(&rhs);
        return BinaryExpression(lval, BinaryOperator::division, rval);
    }
    
    ICmfMeshArray::~ICmfMeshArray(void)
    {
        
    }
}