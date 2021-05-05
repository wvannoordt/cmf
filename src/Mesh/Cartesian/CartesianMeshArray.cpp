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
        deleteMeshBuffer = false;
        isAllocated = false;
        handler = handler_in;
        arrayHandler = handler_in;
        filter = filter_in;
        exchangePattern = NULL;
        rank = info.rank;
        elementType = info.elementType;
        GetDefinedNodes();
        AllocateInitialBlocks();
        CreateExchangePattern();
        this->RegisterToBlocks(handler->mesh->Blocks());
    }
    
    void CartesianMeshArray::AllocateInitialBlocks(void)
    {
        size_t numBlocksToAllocate = definedNodes.size();
        size_t blockSizeInElements = GetArraySizePerBlock();
        deleteMeshBuffer = true;
        meshBuffer = new CartesianMeshBuffer(blockSizeInElements, elementType);
        meshBuffer->ReserveBlocks(numBlocksToAllocate);
        DefinePointerMap();
    }
    
    CartesianMesh* CartesianMeshArray::Mesh(void)
    {
        return this->handler->mesh;
    }
    
    void CartesianMeshArray::CreateExchangePattern()
    {
        exchangePattern = this->handler->GetDefaultExchangeHandler()->CreateMeshArrayExchangePattern(this);
        WriteLine(0, WarningStr() + " CartesianMeshArray::CreateExchangePattern not fully implemented");
    }
    
    void CartesianMeshArray::Exchange(void)
    {
        WriteLine(7, "Exchange \"" + variableName + "\" on mesh \"" + handler->mesh->title + "\"");
        exchangePattern->ExchangeData();
    }
    
    BlockInfo CartesianMeshArray::GetBlockInfo(RefinementTreeNode* node)
    {
        return handler->mesh->GetBlockInfo(node);
    }
    
    void CartesianMeshArray::GetDefinedNodes(void)
    {
        definedNodes.clear();
        for (BlockIterator i(handler->mesh, filter, IterableMode::parallel); i.HasNext(); i++)
        {
            RefinementTreeNode* curNode = i.Node();
            definedNodes.push_back(curNode);
            WriteLine(9, "Define \"" + GetFullName() + "\" on block " + PtrToStr(curNode));
        }
    }
    
    void CartesianMeshArray::OnPostRefinementCallback(std::vector<RefinementTreeNode*>& newChildNodes, std::vector<RefinementTreeNode*> newParentNodes)
    {
        //Could do any of the following:
        //prolongation operation using built-in procedure, with or without guards
        //custom prolongation operation
        //Exchange
        //Etc
        
        //Recompute this list of nodes over which this variable is defined
        this->GetDefinedNodes();
        for (auto p:newParentNodes)
        {
            if (pointerMap.find(p)!=pointerMap.end() && !this->filter(p))
            {
                //Yield the block data to the mesh buffer object if the parent no longer needs it
                meshBuffer->Yield(pointerMap[p]);
                pointerMap.erase(p);
            }
        }
        
        for (auto p:newChildNodes)
        {
            void* newChildBasePtr = meshBuffer->Claim();
            if (pointerMap.find(p)!=pointerMap.end())
            {
                CmfError("Attempted to re-allocate an already-allocated pointer for a new child block after refinement...");
            }
            pointerMap.insert({p, newChildBasePtr});
        }
        
        meshBuffer->ClearVacantChunks();
    }
    
    void* CartesianMeshArray::GetNodePointerWithNullDefault(RefinementTreeNode* node)
    {
        if (pointerMap.find(node)==pointerMap.end()) return NULL;
        return pointerMap[node];
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
        return handler->mesh->Blocks();
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
        for (auto& node:definedNodes)
        {
            void* newPtr = meshBuffer->Claim();
            pointerMap.insert({node, newPtr});
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
        size_t output = numCells * rankMult;
        return output;
    }
    
    CartesianMeshArray::~CartesianMeshArray(void)
    {
        definedNodes.clear();
        if (deleteMeshBuffer)
        {
            delete meshBuffer;
        }
    }
    
    void CartesianMeshArray::Destroy(void)
    {
        WriteLine(4, "Destroying variable \"" + variableName + "\" on mesh \"" + handler->mesh->title + "\"");
    }
    
    void CartesianMeshArray::GetParallelDataBuffer(ParallelDataBuffer& buf)
    {
        size_t currentOffset = 0;
        size_t blockSizeBytes = meshBuffer->BlockSizeBytes();
        for (BlockIterator iter(this, this->filter, IterableMode::serial); iter.HasNext(); iter++)
        {
            auto node = iter.Node();
            if (pointerMap.find(node) != pointerMap.end())
            {
                char* ptr = (char*)pointerMap[node];
                buf.Add<char>(ptr, blockSizeBytes, currentOffset);
            }
            currentOffset += blockSizeBytes;
        }
    }
    
    void CartesianMeshArray::WriteFilterToFile(ParallelFile& file)
    {
        int p = 0;
        for (BlockIterator iter(this->GetRefinementBlockObject(), BlockFilters::Every, IterableMode::serial); iter.HasNext(); iter++)
        {
            if (filter(iter.Node()))
            {
                file.Write(p);
            }
            p++;
        }
    }
    
    void CartesianMeshArray::ReadFromFile(ParallelFile& file)
    {
        
    }
    
    void CartesianMeshArray::WriteToFile(ParallelFile& file)
    {
        file.Write("CartesianMeshArray");
        file.Write(strformat("mesh: {}", handler->mesh->GetTitle()));
        file.Write(strformat("name: {}", variableName));
        file.Write(strformat("type: {}", CmfArrayTypeToString(elementType)));
        file.Write("<components>");
        for (auto comp: variableComponentNames)
        {
            file.Write(comp);
        }
        file.Write("</components>");
        file.Write(strformat("rank: {}", rank));
        file.Write("<dims>");
        for (auto dim: dims)
        {
            file.Write(strformat("{}", dim));
        }
        file.Write("</dims>");
        file.Write("<nodes>");
        this->WriteFilterToFile(file);
        file.Write("</nodes>");
        ParallelDataBuffer parallelBuf;
        this->GetParallelDataBuffer(parallelBuf);
        file.Write("<data>");
        file.ParallelWrite(parallelBuf);
        file.Write("</data>");
    }
}