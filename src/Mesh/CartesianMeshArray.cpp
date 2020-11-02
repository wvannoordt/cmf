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
        Allocate();
        GetDefinedNodes();
        DefinePointerMap();
    }
    
    void CartesianMeshArray::Allocate(void)
    {
        size_t numBlocksToAllocate = handler->mesh->Blocks()->Size(filter);
        size_t totalAllocSize = numBlocksToAllocate*GetArraySizePerBlock();
        WriteLine(4, 
            "Defining variable \"" + variableName + "\" on mesh \"" + 
            handler->mesh->title + "\" over " +
            std::to_string(numBlocksToAllocate) + " blocks. Total size: " +
            NiceCommaString(totalAllocSize));
        ptr = Cmf_Alloc(totalAllocSize);
    }
    
    void CartesianMeshArray::GetDefinedNodes(void)
    {
        for (BlockIterator i(handler->mesh->Blocks(), filter); i.HasNext(); i++)
        {
            RefinementTreeNode* curNode = i.Node();
            definedNodes.push_back(curNode);
            WriteLine(8, "Define \"" + GetFullName() + "\" on block " + PtrToStr(curNode));
        }
    }
    
    std::string CartesianMeshArray::GetFullName(void)
    {
        return handler->mesh->title + "::" + variableName;
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
        for (int d = 0; d < CMF_DIM; d++) numCells *= handler->mesh->meshDataDim[d];
        return numCells * elementSize;
    }
    
    CartesianMeshArray::~CartesianMeshArray(void)
    {
        definedNodes.clear();
    }
    
    void CartesianMeshArray::Destroy(void)
    {
        WriteLine(4, "Destroying variable \"" + variableName + "\" + on mesh \"" + handler->mesh->title + "\"");
        Cmf_Free(ptr);
    }
}