#include "CartesianMeshArray.h"
#include "DebugTools.hx"
#include "CartesianMeshArrayHandler.h"
#include "CartesianMesh.h"
#include "StringUtils.h"
#include "BlockIterator.h"
#include "CartesianMeshArrayParallelVtkWriter.h"
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
        GetDefinedNodesAndAllocatedNodes();
        AllocateInitialBlocks();
        CreateExchangePattern();
        this->RegisterToBlocks(handler->mesh->Blocks());
    }
    
    void CartesianMeshArray::AllocateInitialBlocks(void)
    {
        size_t numBlocksToAllocate = allocatedNodes.size();
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
    
    void CartesianMeshArray::GetDefinedNodesAndAllocatedNodes(void)
    {
        definedNodes.clear();
        for (BlockIterator i(handler->mesh, filter, IterableMode::serial); i.HasNext(); i++)
        {
            RefinementTreeNode* curNode = i.Node();
            definedNodes.push_back(curNode);
        }
        allocatedNodes.clear();
        for (BlockIterator i(handler->mesh, filter, IterableMode::parallel); i.HasNext(); i++)
        {
            RefinementTreeNode* curNode = i.Node();
            allocatedNodes.push_back(curNode);
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
        this->GetDefinedNodesAndAllocatedNodes();
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
            if (this->ParallelPartitionContainsNode(p))
            {
                void* newChildBasePtr = meshBuffer->Claim();
                if (pointerMap.find(p)!=pointerMap.end())
                {
                    CmfError("Attempted to re-allocate an already-allocated pointer for a new child block after refinement...");
                }
                pointerMap.insert({p, newChildBasePtr});
            }
        }
        
        meshBuffer->ClearVacantChunks();
        
        //Redefine the exchange pattern (there is definitely a better way to do this)
        exchangePattern = this->handler->defaultExchangeHandler->CreateMeshArrayExchangePattern(this);
    }
    
    void* CartesianMeshArray::GetNodePointerWithNullDefault(RefinementTreeNode* node)
    {
        if (pointerMap.find(node)==pointerMap.end()) return NULL;
        return pointerMap[node];
    }
    
    std::vector<RefinementTreeNode*>::iterator CartesianMeshArray::begin() noexcept
    {
        return allocatedNodes.begin();
    }
    
    std::vector<RefinementTreeNode*>::const_iterator CartesianMeshArray::begin() const noexcept
    {
        return allocatedNodes.begin();
    }
    
    std::vector<RefinementTreeNode*>::iterator CartesianMeshArray::end() noexcept
    {
        return allocatedNodes.end();
    }
    
    std::vector<RefinementTreeNode*>::const_iterator CartesianMeshArray::end() const noexcept
    {
        return allocatedNodes.end();
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
    
    size_t CartesianMeshArray::GetSingleCellSizeBytes(void)
    {
        size_t output = SizeOfArrayType(elementType);
        int rankMult = 1;
        for (int i = 0; i < rank; i++) rankMult *= dims[i];
        return output * rankMult;
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
        for (auto& node:allocatedNodes)
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
        allocatedNodes.clear();
        if (deleteMeshBuffer)
        {
            delete meshBuffer;
        }
    }
    
    void CartesianMeshArray::ExportFile(std::string directory, std::string fileTitle)
    {
        CartesianMeshArrayParallelVtkWriter writerObj(directory, fileTitle);
        writerObj.Export(*this);
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
    
    bool CartesianMeshArray::ParallelPartitionContainsNode(RefinementTreeNode* node)
    {
        return this->Mesh()->GetPartition()->Mine(node);
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
    
    void CartesianMeshArray::VerifyFilterFromFile(ParallelFile& file)
    {
        int p = 0;
        for (BlockIterator iter(this->GetRefinementBlockObject(), BlockFilters::Every, IterableMode::serial); iter.HasNext(); iter++)
        {
            if (filter(iter.Node()))
            {
                int readNode = -1;
                strunformat(file.Read(), "{}", readNode);
                if (p != readNode)
                {
                    CmfError(strformat("VerifyFilterFromFile filter compatibility error: expecting node {}, found node {} in file \"{}\"", p, readNode, file.OpenFileName()));
                }
            }
            p++;
        }
    }
    
    void CartesianMeshArray::ReadFromFile(ParallelFile& file)
    {
        std::string synchErrorStr = "CartesianMeshArray::ReadFromFile synchronization error: expecting token \"{}\", but found \"{}\" in file " + file.OpenFileName();
        std::string compatErrorStr = "CartesianMeshArray::ReadFromFile compatibility error: expecting value of \"{}\" for \"{}\", but found \"{}\" in file " + file.OpenFileName();
        std::string line;
        if ((line=file.Read())!="CartesianMeshArray") CmfError(strformat(synchErrorStr, "CartesianMeshArray", line));
        line = file.Read();
        line = file.Read();
        std::string readType;
        strunformat(line=file.Read(), "type: {}", readType);
        if (readType != CmfArrayTypeToString(elementType)) CmfError(strformat(synchErrorStr, CmfArrayTypeToString(elementType), readType));
        
        int readRank;
        strunformat(line=file.Read(), "rank: {}", readRank);
        if (rank != readRank) CmfError(strformat(compatErrorStr, rank, "rank", readRank));
        if ((line=file.Read())!="<dims>") CmfError(strformat(synchErrorStr, "<dims>", line));
        for (int i = 0; i < readRank; i++)
        {
            int readDim = -1;
            strunformat(line=file.Read(), "{}", readDim);
            if (dims[i] != readDim) CmfError(strformat(compatErrorStr, dims[i], strformat("dims[{}]", i), readDim));
        }
        if ((line=file.Read())!="</dims>") CmfError(strformat(synchErrorStr, "</dims>", line));
        if ((line=file.Read())!="<components>") CmfError(strformat(synchErrorStr, "<components>", line));
        for (auto comp: variableComponentNames)
        {
            line = file.Read();
        }
        if ((line=file.Read())!="</components>") CmfError(strformat(synchErrorStr, "</components>", line));
        
        if ((line=file.Read())!="<nodes>") CmfError(strformat(synchErrorStr, "<nodes>", line));
        this->VerifyFilterFromFile(file);
        if ((line=file.Read())!="</nodes>") CmfError(strformat(synchErrorStr, "</nodes>", line));
        
        ParallelDataBuffer parallelBuf;
        this->GetParallelDataBuffer(parallelBuf);
        if ((line=file.Read())!="<data>") CmfError(strformat(synchErrorStr, "<data>", line));
        file.ParallelRead(parallelBuf);
        if ((line=file.Read())!="</data>") CmfError(strformat(synchErrorStr, "</data>", line));
    }
    
    void CartesianMeshArray::WriteToFile(ParallelFile& file)
    {
        file.Write("CartesianMeshArray");
        file.Write(strformat("mesh: {}", handler->mesh->GetTitle()));
        file.Write(strformat("name: {}", variableName));
        file.Write(strformat("type: {}", CmfArrayTypeToString(elementType)));
        file.Write(strformat("rank: {}", rank));
        file.Write("<dims>");
        for (auto dim: dims)
        {
            file.Write(strformat("{}", dim));
        }
        file.Write("</dims>");
        file.Write("<components>");
        for (auto comp: variableComponentNames)
        {
            file.Write(comp);
        }
        file.Write("</components>");
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