#include "CartesianMeshArray.h"
#include "DebugTools.hx"
#include "CartesianMeshArrayHandler.h"
#include "CartesianMesh.h"
#include "StringUtils.h"
#include "BlockIterator.h"
namespace cmf
{
    CartesianMeshArray::CartesianMeshArray(ArrayInfo info, CartesianMeshArrayHandler* handler_in, NodeFilter_t filter_in) : ICmfMeshArray(info)
    {
        isAllocated = false;
        handler = handler_in;
        filter = filter_in;
        exchangePattern = NULL;
        rank = info.rank;
        arrayDimensions.resize(info.rank, 0);
        for (int i = 0; i < info.rank; i++)
        {
            arrayDimensions[i] = info.dimensions[i];
        }
        elementSize = info.elementSize;
        GetDefinedNodes();
        Allocate();
        DefinePointerMap();
        CreateExchangePattern();
    }
    
    void CartesianMeshArray::Allocate(void)
    {
        size_t numBlocksToAllocate = definedNodes.size();
        size_t totalAllocSize = numBlocksToAllocate*GetArraySizePerBlock();
        WriteLine(4, 
            "Defining variable \"" + variableName + "\" on mesh \"" + 
            handler->mesh->title + "\" over " +
            std::to_string(numBlocksToAllocate) + " blocks. Total size: " +
            NiceCommaString(totalAllocSize));
        ptr = Cmf_Alloc(totalAllocSize);
    }
    
    void CartesianMeshArray::CreateExchangePattern()
    {
        exchangePattern = this->handler->GetDefaultExchangeHandler()->CreateMeshArrayExchangePattern(this);
        WriteLine(0, "WARNING: CartesianMeshArray::CreateExchangePattern not fully implemented");
    }
    
    void CartesianMeshArray::Exchange(void)
    {
        WriteLine(7, "Exchange \"" + variableName + "\" on mesh \"" + handler->mesh->title + "\"");
        exchangePattern->ExchangeData();
    }
    
    void CartesianMeshArray::GetDefinedNodes(void)
    {
        for (BlockIterator i(handler->mesh, filter, IterableMode::parallel); i.HasNext(); i++)
        {
            RefinementTreeNode* curNode = i.Node();
            definedNodes.push_back(curNode);
            WriteLine(9, "Define \"" + GetFullName() + "\" on block " + PtrToStr(curNode));
        }
    }
    
    std::vector<RefinementTreeNode*>::iterator CartesianMeshArray::begin() noexcept
    {
        return definedNodes.begin();
    }
    
    std::vector<RefinementTreeNode*>::const_iterator CartesianMeshArray::begin() const noexcept
    {
        return definedNodes.begin();
    }
    
    std::vector<RefinementTreeNode*>::iterator CartesianMeshArray::end() noexcept
    {
        return definedNodes.end();
    }
    
    std::vector<RefinementTreeNode*>::const_iterator CartesianMeshArray::end() const noexcept
    {
        return definedNodes.end();
    }
    
    NodeFilter_t CartesianMeshArray::GetFilter(void)
    {
        return filter;
    }
    
    RefinementBlock* CartesianMeshArray::GetRefinementBlockObject(void)
    {
        handler->mesh->Blocks();
    }
    
    std::string CartesianMeshArray::GetFullName(void)
    {
        return handler->mesh->title + "_" + variableName;
    }
    
    std::vector<RefinementTreeNode*>* CartesianMeshArray::GetAllNodes(void)
    {
        return &definedNodes;
    }
    
    CartesianMeshArrayPointerPair CartesianMeshArray::operator [] (BlockIterator& it)
    {
        return (*this)[it.Node()];
    }
    
    CartesianMeshArrayPointerPair CartesianMeshArray::operator [] (RefinementTreeNode* node)
    {
        if (!IsSupportedBlock(node)) CmfError("Attempted to index variable \"" + variableName + "\" on mesh \"" + handler->mesh->title + "\" on an unsupported block");
        CartesianMeshArrayPointerPair output;
        output.array = this;
        output.pointer = pointerMap[node];
        return output;
    }
    
    bool CartesianMeshArray::IsSupportedBlock(RefinementTreeNode* node)
    {
        return (pointerMap.find(node)!=pointerMap.end());
    }
    
    void CartesianMeshArray::DefinePointerMap(void)
    {
        size_t pitch = GetArraySizePerBlock();
        for (size_t i = 0; i < definedNodes.size(); i++)
        {
            pointerMap.insert({definedNodes[i],(void*)((char*)ptr + pitch*i)});
        }
    }
    
    size_t CartesianMeshArray::Size(void)
    {
        return definedNodes.size();
    }
    
    size_t CartesianMeshArray::GetArraySizePerBlock(void)
    {
        size_t numCells = 1;
        int rankMult = 1;
        for (int i = 0; i < rank; i++) rankMult *= dims[i];
        for (int d = 0; d < CMF_DIM; d++) numCells *= (handler->mesh->meshDataDim[d] + 2*handler->mesh->exchangeDim[d]);
        return numCells * elementSize * rankMult;
    }
    
    CartesianMeshArray::~CartesianMeshArray(void)
    {
        definedNodes.clear();
    }
    
    void CartesianMeshArray::Destroy(void)
    {
        WriteLine(4, "Destroying variable \"" + variableName + "\" on mesh \"" + handler->mesh->title + "\"");
        Cmf_Free(ptr);
    }
}