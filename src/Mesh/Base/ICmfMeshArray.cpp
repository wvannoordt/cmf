#include "ICmfMeshArray.h"
#include "CmfScreen.h"
#include "MeshArrayExpression.h"
#include "BinaryExpression.h"
#include "Utils.hx"
#include "CmfPrint.h"
#include "StringUtils.h"
#include "ICmfMeshArrayHandler.h"
#include "ICmfMesh.h"
#include "CmfDataBase.h"
namespace cmf
{
    ICmfMeshArray::ICmfMeshArray(ArrayInfo info)
    {
        variableName = info.name;
        if (info.rank > MAX_RANK) CmfError("Rank of variable \"" + variableName + "\" exceeds MAX_RANK (" + std::to_string(MAX_RANK) + "): recompile with greater limit.");
        rank = info.rank;
        elementSize = info.elementSize;
        for (int i = 0; i < rank; i++)
        {
            dims.push_back(info.dimensions[i]);
        }
        SetDefaultComponentNames();
    }
    
    std::string& ICmfMeshArray::ComponentName(void)
    {
        return ComponentName({});
    }
    
    void ICmfMeshArray::SetRequiredPrereqtuisiteDataBaseObjects(void)
    {
        objectsRequiredBeforeAddingToDataBase.Add(arrayHandler);
        objectsRequiredBeforeAddingToDataBase.Add(arrayHandler->Mesh());
    }
    
    void ICmfMeshArray::SetAutomaticallyAddedObjects(void)
    {
        
    }
    
    ParallelGroup* ICmfMeshArray::GetDatabaseParallelGroup(void)
    {
        return this->arrayHandler->Mesh()->GetGroup();
    }
    
    std::string ICmfMeshArray::DataBaseName(void)
    {
        return strformat("{}{}{}", arrayHandler->Mesh()->GetTitle(), CmfDataBase::GetDataBaseDlimiter(), variableName);
    }
    
    std::string& ICmfMeshArray::ComponentName(std::initializer_list<int> index)
    {
        if (index.size() != rank)
        {
            CmfError(strformat("Attempted to get ComponentName for \"{}\" using indexer of rank {}, but rank is {}", variableName, index.size(), rank));
        }
        int i = 0;
        for (auto indexVal:index)
        {
            if (indexVal<0 || indexVal >= dims[i])
            {
                CmfError(strformat("Attempted to get ComponentName for \"{}\", but index {} has value {}, which is outside expected range [0, {})", variableName, i, indexVal, dims[i]));
            }
            i++;
        }
        std::vector<int> indexV;
        for (auto i:index) indexV.push_back(i);
        int idx = IndexNDToIndex1D(indexV, dims);
        return variableComponentNames[idx];
    }
    
    void ICmfMeshArray::SetDefaultComponentNames(void)
    {
        hasInitializedVariableComponentNames = true;
        int numComponentsTotal = 1;
        for (int i = 0; i < rank; i++) numComponentsTotal *= dims[i];
        variableComponentNames.resize(numComponentsTotal);
        if (rank==0)
        {
            variableComponentNames[0] = "data";
        }
        else
        {            
            for (int i = 0; i < numComponentsTotal; i++)
            {
                std::vector<int> indices = Index1DToIndexND(i, dims);
                variableComponentNames[i] = GetDefaultComponentName(indices);
            }
        }
    }
    
    std::string ICmfMeshArray::GetDefaultComponentName(std::vector<int>& index)
    {
        std::string output = "data";
        for (auto i:index)
        {
            output+= "_";
            output+= ZFill(i, 2);
        }
        return output;
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