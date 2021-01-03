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
        GetDefinedNodes();
        Allocate();
        DefinePointerMap();
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
    
    void CartesianMeshArray::Exchange(void)
    {
        WriteLine(7, "Exchange \"" + variableName + "\" on mesh \"" + handler->mesh->title + "\"");
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
    
    void* CartesianMeshArray::operator [] (BlockIterator& it)
    {
        if (!IsSupportedBlock(it.Node())) CmfError("Attempted to index variable \"" + variableName + "\" on mesh \"" + handler->mesh->title + "\" on an unsupported block");
        return pointerMap[it.Node()];
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