#include "MultiTransaction.h"
#include "CmfError.h"
#include <cstring>

namespace cmf
{
    MultiTransaction::MultiTransaction(void* target_in, std::vector<size_t> offsets_in, std::vector<size_t> sizes_in, int sender_in, int receiver_in)
        : IDataTransaction(sender_in, receiver_in)
    {
        target  = target_in;
        offsets = offsets_in;
        sizes   = sizes_in;
        if (offsets.size() != sizes.size())
        {
            CmfError("A MultiTransaction has been created with inconsistent offsets and sizes. Found "
                + std::to_string(offsets.size()) + " offsets, but " + std::to_string(sizes.size()) + " sizes.");
        }
        packedSize = 0;
        for (const auto s:sizes) packedSize += s;
    }
    
    MultiTransaction::~MultiTransaction(void)
    {
        
    }
    
    size_t MultiTransaction::GetPackedSize(void)
    {
        return packedSize;
    }
    
    void MultiTransaction::Pack(char* buf)
    {
        char* copyTo = buf;
        char* cTarget = (char*)target;
        int numOffsets = offsets.size();
        for (int i = 0; i < numOffsets; i++)
        {
            memcpy(copyTo, cTarget + offsets[i], sizes[i]);
            copyTo += sizes[i];
        }
    }
    
    void MultiTransaction::Unpack(char* buf)
    {
        char* copyFrom = buf;
        char* cTarget = (char*)target;
        int numOffsets = offsets.size();
        for (int i = 0; i < numOffsets; i++)
        {
            memcpy(cTarget + offsets[i], copyFrom, sizes[i]);
            copyFrom += sizes[i];
        }
    }
}