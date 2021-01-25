#include "SingleTransaction.h"
#include <iostream>
#include <cstring>
#include "ParallelGroup.h"

namespace cmf
{
    SingleTransaction::SingleTransaction(void* target_in, size_t size_in, int sender_in, int receiver_in)
        : IDataTransaction(sender_in, receiver_in)
    {
        target = target_in;
        size = size_in;
    }
    
    size_t SingleTransaction::GetPackedSize(void)
    {
        return size;
    }
    
    void SingleTransaction::Pack(char* buf)
    {
        char* copyFrom = (char*)target;
        memcpy(buf, copyFrom, size);
    }
    
    void SingleTransaction::Unpack(char* buf)
    {
        char* copyTo = (char*)target;
        memcpy(copyTo, buf, size);
    }

    SingleTransaction::~SingleTransaction(void)
    {
        
    }
}