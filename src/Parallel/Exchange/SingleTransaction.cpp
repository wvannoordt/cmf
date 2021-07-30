#include "SingleTransaction.h"
#include <iostream>
#include <cstring>
#include "ParallelGroup.h"

namespace cmf
{
    SingleTransaction::SingleTransaction(void* sendTarget_in, void* recvTarget_in, size_t size_in, ComputeDevice sender_in, ComputeDevice receiver_in)
        : IDataTransaction(sender_in, receiver_in)
    {
        sendTarget = sendTarget_in;
        recvTarget = recvTarget_in;
        size = size_in;
    }
    
    size_t SingleTransaction::GetPackedSize(void)
    {
        return size;
    }
    
    void SingleTransaction::Pack(char* buf)
    {
        char* copyFrom = (char*)sendTarget;
        memcpy(buf, copyFrom, size);
    }
    
    void SingleTransaction::Unpack(char* buf)
    {
        char* copyTo = (char*)recvTarget;
        memcpy(copyTo, buf, size);
    }

    SingleTransaction::~SingleTransaction(void)
    {
        
    }
}