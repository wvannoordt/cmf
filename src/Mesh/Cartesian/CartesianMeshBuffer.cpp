#include "CartesianMeshBuffer.h"
#include "CmfGC.h"
#include "CmfError.h"
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
        CartesianDataChunk newChunk;
        newChunk.base = Cmf_Alloc(numBlocks*blockArraySize*SizeOfArrayType(arrayType));
        newChunk.numBlocks = numBlocks;
        chunks.push_back(newChunk);
    }
    
    void CartesianMeshBuffer::Clear(void)
    {
        for (auto ch:chunks)
        {
            Cmf_Free(ch.base);
        }
    }
    
    size_t CartesianMeshBuffer::BlockSizeBytes(void)
    {
        return blockArraySize*SizeOfArrayType(arrayType);
    }
    
    void CartesianMeshBuffer::Yield(void* ptr)
    {
        
    }
    
    void* CartesianMeshBuffer::Claim(void)
    {
        void* output = (void*)((char*)chunks[0].base + tempCounter*blockArraySize);
        tempCounter++;
        return output;
    }
}