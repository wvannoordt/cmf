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
        CartesianDataChunk* newChunk = new CartesianDataChunk();
        newChunk->base = Cmf_Alloc(numBlocks*blockArraySize*SizeOfArrayType(arrayType));
        newChunk->numBlocks = numBlocks;
        newChunk->numberOfVacantBlocks = numBlocks;
        chunks.push_back(newChunk);
        for (size_t i = 0; i < numBlocks; i++)
        {
            size_t offset = i*blockArraySize*SizeOfArrayType(arrayType);
            void* blockPointer = (void*)((char*)newChunk->base + offset);
            pointerToChunks.insert({blockPointer, newChunk});
            vacantBlocks.push_back({blockPointer, newChunk});
        }
    }
    
    void CartesianMeshBuffer::Clear(void)
    {
        for (auto& ch:chunks)
        {
            if (ch->base != NULL)
            {
                Cmf_Free(ch->base);
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
        vacantBlocks.push_back({ptr, chunk});
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
                Cmf_Free(ch->base);
                ch->base = NULL;
                ch->numBlocks = 0;
                ch->numberOfVacantBlocks = 0;
            }
            numVacantBlocks += ch->numberOfVacantBlocks;
        }
        
        WriteLine(4, strformat("Cleared {} blocks from {} chunks. Available blocks remaining: {}", numBlocksCleared, numChunksCleared, numVacantBlocks));
    }
    
    void* CartesianMeshBuffer::Claim(void)
    {
        if (vacantBlocks.empty())
        {
            WriteLine(4, strformat("Reserving {} Cartesian blocks", blockBatchSize));
            this->ReserveBlocks(blockBatchSize);
        }
        
        
        void* output = vacantBlocks.front().first;
        CartesianDataChunk* chunk = vacantBlocks.front().second;
        
        chunk->numberOfVacantBlocks--;
        
        vacantBlocks.pop_front();
        return output;
    }
}