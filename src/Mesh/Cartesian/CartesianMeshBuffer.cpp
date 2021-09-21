#include "CartesianMeshBuffer.h"
#include "CmfGC.h"
#include "CmfError.h"
#include "StringUtils.h"
#include "CmfScreen.h"
namespace cmf
{
    CartesianMeshBuffer::CartesianMeshBuffer(size_t blockArraySize_in, CmfArrayType::CmfArrayType arrayType_in)
    {
        blockArraySize = blockArraySize_in;
        arrayType = arrayType_in;
        tempCounter = 0;
        blockBatchSize = 10;
    }

    CartesianMeshBuffer::~CartesianMeshBuffer(void)
    {
        Clear();
    }
    
    void CartesianMeshBuffer::ReserveBlocks(int numBlocks)
    {
        this->ReserveBlocks(numBlocks, MemSpace::Cpu, 0);
    }
    
    void CartesianMeshBuffer::ReserveBlocks(int numBlocks, MemSpace::MemSpace space, int gpuDeviceId)
    {
        if (numBlocks==0) return;
        if (numBlocks<0) CmfError("CartesianMeshBuffer::ReserveBlocks requesting to allocate negative memory!");
        CartesianDataChunk* newChunk = new CartesianDataChunk();
        if (space==MemSpace::Gpu)
        {
            newChunk->base = Cmf_GpuAlloc(numBlocks*blockArraySize*SizeOfArrayType(arrayType), gpuDeviceId);
        }
        else
        {
            newChunk->base = Cmf_Alloc(numBlocks*blockArraySize*SizeOfArrayType(arrayType));
        }
        newChunk->numBlocks = numBlocks;
        newChunk->numberOfVacantBlocks = numBlocks;
        chunks.push_back(newChunk);
        for (size_t i = 0; i < numBlocks; i++)
        {
            size_t offset = i*blockArraySize*SizeOfArrayType(arrayType);
            void* blockPointer = (void*)((char*)newChunk->base + offset);
            pointerToChunks.insert({blockPointer, newChunk});
            if (space==MemSpace::Gpu) {vacantBlocksGpu.push_back({blockPointer, newChunk});}
            else {vacantBlocksCpu.push_back({blockPointer, newChunk});}
        }
    }
    
    void CartesianMeshBuffer::Clear(void)
    {
        for (auto& ch:chunks)
        {
            if (ch->base != NULL)
            {
                if (ch->gpu)
                {
                    Cmf_GpuFree(ch->base);
                }
                else
                {
                    Cmf_Free(ch->base);
                }
            }
            delete ch;
        }
    }
    
    size_t CartesianMeshBuffer::BlockSizeBytes(void)
    {
        return blockArraySize*SizeOfArrayType(arrayType);
    }
    
    void CartesianMeshBuffer::Yield(void* ptr)
    {
        if (pointerToChunks.find(ptr)==pointerToChunks.end())
        {
            CmfError("Attempted to yield an unmanaged buffer to CartesianMeshBuffer. This is a bad one...");
        }
        CartesianDataChunk* chunk = pointerToChunks[ptr];
        chunk->numberOfVacantBlocks++;
        if (chunk->gpu) {vacantBlocksGpu.push_back({ptr, chunk});}
        else {vacantBlocksCpu.push_back({ptr, chunk});}
    }
    
    void CartesianMeshBuffer::ClearVacantChunks(void)
    {
        int numChunksCleared = 0;
        int numBlocksCleared = 0;
        int numVacantBlocks = 0;
        
        //should I delete the chunks? for now, they are just freed and reset
        for (auto& ch:chunks)
        {
            if (ch->numBlocks == ch->numberOfVacantBlocks)
            {
                numChunksCleared++;
                numBlocksCleared += ch->numBlocks;
                if (ch->gpu) {Cmf_GpuFree(ch->base);}
                else {Cmf_Free(ch->base);}
                ch->base = NULL;
                ch->numBlocks = 0;
                ch->numberOfVacantBlocks = 0;
            }
            numVacantBlocks += ch->numberOfVacantBlocks;
        }
        
        WriteLine(4, strformat("Cleared {} blocks from {} chunks. Available blocks remaining: {}", numBlocksCleared, numChunksCleared, numVacantBlocks));
    }
    
    void* CartesianMeshBuffer::Claim()
    {
        return this->Claim(MemSpace::Cpu, 0);
    }
    
    void* CartesianMeshBuffer::Claim(MemSpace::MemSpace space, int gpuDeviceId)
    {
        auto relevantBlocks = &vacantBlocksCpu;
        if (space == MemSpace::Gpu) relevantBlocks = &vacantBlocksGpu;
        if (space == MemSpace::Gpu && vacantBlocksGpu.empty())
        {
            WriteLine(4, strformat("Reserving {} Cartesian blocks (GPU)", blockBatchSize));
            this->ReserveBlocks(blockBatchSize, space, gpuDeviceId);
        }
        if (space == MemSpace::Cpu && vacantBlocksCpu.empty())
        {
            WriteLine(4, strformat("Reserving {} Cartesian blocks", blockBatchSize));
            this->ReserveBlocks(blockBatchSize);
        }
        
        
        void* output = relevantBlocks->front().first;
        CartesianDataChunk* chunk = relevantBlocks->front().second;
        
        chunk->numberOfVacantBlocks--;
        
        relevantBlocks->pop_front();
        return output;
    }
}