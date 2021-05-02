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
        void* newPtr = Cmf_Alloc(numBlocks*blockArraySize*SizeOfArrayType(arrayType));
        pointers.push_back(newPtr);
    }
    
    void CartesianMeshBuffer::Clear(void)
    {
        for (auto p:pointers)
        {
            Cmf_Free(p);
        }
    }
    
    size_t CartesianMeshBuffer::BlockSizeBytes(void)
    {
        return blockArraySize*SizeOfArrayType(arrayType);
    }
    
    void* CartesianMeshBuffer::Claim(void)
    {
        void* output = (void*)((char*)pointers[0] + tempCounter*blockArraySize);
        tempCounter++;
        return output;
    }
}