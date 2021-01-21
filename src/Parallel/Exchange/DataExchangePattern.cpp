#include "DataExchangePattern.h"
#include "CmfScreen.h"
#include "CmfGC.h"
namespace cmf
{
    DataExchangePattern::DataExchangePattern(ParallelGroup* group_in)
    {
        group = group_in;
        resizeOutBufferRequired = true;
        resizeInBufferRequired = true;
        sendBufferIsAllocated = false;
        receiveBufferIsAllocated = false;
        sendBuffer = NULL;
        receiveBuffer = NULL;
    }
    
    DataExchangePattern::~DataExchangePattern(void)
    {
        if (sendBufferIsAllocated)
        {
            sendBufferIsAllocated = false;
            Cmf_Free(sendBuffer);
        }
        if (receiveBufferIsAllocated)
        {
            receiveBufferIsAllocated = false;
            Cmf_Free(receiveBuffer);
        }
        for (auto tr:transactions)
        {
            delete tr;
        }
    }
    
    void DataExchangePattern::Add(IDataTransaction* transaction)
    {
        transactions.push_back(transaction);
    }
    
    void DataExchangePattern::Pack(void)
    {
        char* ptr = sendBuffer;
        for (const auto tr:transactions)
        {
            if (tr->Sender() == group->Rank())
            {
                tr->Pack(ptr);
                ptr += tr->GetPackedSize();
            }
        }
    }
    
    void DataExchangePattern::Unpack(void)
    {
        char* ptr = receiveBuffer;
        for (const auto tr:transactions)
        {
            if (tr->Receiver() == group->Rank())
            {
                tr->Unpack(ptr);
                ptr += tr->GetPackedSize();
            }
        }
    }
    
    void DataExchangePattern::ExchangeData(void)
    {
        if (resizeOutBufferRequired || !sendBufferIsAllocated)    this->ResizeOutBuffer();
        if (resizeInBufferRequired  || !receiveBufferIsAllocated) this->ResizeInBuffer();
        Pack();
        WriteLine(0, "WARNING: DataExchangePattern::ExchangeData not fully implemented");
        Unpack();
    }
    
    //Considering using a predicate here...
    void DataExchangePattern::ResizeOutBuffer(void)
    {
        size_t totalSize = 0;
        for (const auto tr:transactions)
        {
            if (tr->Sender() == group->Rank()) totalSize += tr->GetPackedSize();
        }
        // Consider a wrapper for realloc() if downsizing, it is a lot faster!
        if (sendBufferIsAllocated) Cmf_Free(sendBuffer);
        sendBuffer = (char*)Cmf_Alloc(totalSize);
    }
    
    void DataExchangePattern::ResizeInBuffer(void)
    {
        size_t totalSize = 0;
        for (const auto tr:transactions)
        {
            if (tr->Receiver() == group->Rank()) totalSize += tr->GetPackedSize();
        }
        // Consider a wrapper for realloc() if downsizing, it is a lot faster!
        if (receiveBufferIsAllocated) Cmf_Free(receiveBuffer);
        receiveBuffer = (char*)Cmf_Alloc(totalSize);
    }
}