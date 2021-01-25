#include "DataExchangePattern.h"
#include "CmfScreen.h"
#include "CmfGC.h"
namespace cmf
{
    DataExchangePattern::DataExchangePattern(ParallelGroup* group_in)
    {
        group = group_in;
        int groupSize = group->Size();
        resizeOutBufferRequired.resize(groupSize, true);
        resizeInBufferRequired.resize(groupSize, true);
        sendBufferIsAllocated.resize(groupSize, false);
        receiveBufferIsAllocated.resize(groupSize, false);
        sendBuffer.resize(groupSize, NULL);
        receiveBuffer.resize(groupSize, NULL);
        requestHandles.resize(groupSize);
        sendSizes.resize(groupSize, 0);
        receiveSizes.resize(groupSize, 0);
    }
    
    DataExchangePattern::~DataExchangePattern(void)
    {
        for (int rank = 0; rank < group->Size(); rank++)
        {
            if (sendBufferIsAllocated[rank])
            {
                sendBufferIsAllocated[rank] = false;
                Cmf_Free(sendBuffer[rank]);
            }
            if (receiveBufferIsAllocated[rank])
            {
                receiveBufferIsAllocated[rank] = false;
                Cmf_Free(receiveBuffer[rank]);
            }
        }
        for (auto tr:transactions)
        {
            delete tr;
        }
    }
    
    void DataExchangePattern::Add(IDataTransaction* transaction)
    {
        int sender = transaction->Sender();
        int receiver = transaction->Receiver();
        int currentRank = group->Rank();
        if ((sender == currentRank) || (receiver == currentRank))
        {
            transactions.push_back(transaction);
            if ((sender == currentRank) && (transaction->GetPackedSize() > 0))
            {
                resizeOutBufferRequired[receiver] = true;
                sendSizes[currentRank] += transaction->GetPackedSize();
            }
            if ((receiver == currentRank) && (transaction->GetPackedSize() > 0))
            {
                resizeInBufferRequired[sender] = true;
                receiveSizes[currentRank] += transaction->GetPackedSize();
            }
        }
    }
    
    void DataExchangePattern::Pack(void)
    {
        pointerIndices = sendBuffer; //Vector deep-copy.
        for (const auto tr:transactions)
        {
            int currentRank = group->Rank();
            if (tr->Sender() == currentRank)
            {
                tr->Pack(pointerIndices[currentRank]);
                pointerIndices[currentRank] += tr->GetPackedSize();
            }
        }
    }
    
    void DataExchangePattern::Unpack(void)
    {
        pointerIndices = receiveBuffer; //Vector deep-copy.
        
        //Note that self-to-self transactions are not copied between
        //send and receive buffers on the same rank. Why would they be? :-)
        pointerIndices[group->Rank()] = sendBuffer[group->Rank()];
        
        for (const auto tr:transactions)
        {
            int currentRank = group->Rank();
            if (tr->Receiver() == currentRank)
            {
                tr->Unpack(pointerIndices[currentRank]);
                pointerIndices[currentRank] += tr->GetPackedSize();
            }
        }
    }
    
    void DataExchangePattern::ExchangeData(void)
    {
        for (int rank = 0; rank < group->Size(); rank++)
        {
            if (resizeOutBufferRequired[rank] || !sendBufferIsAllocated[rank])    this->ResizeOutBuffer(rank);
            if (resizeInBufferRequired[rank]  || !receiveBufferIsAllocated[rank]) this->ResizeInBuffer(rank);
        }
        //Aggregate the data
        Pack();
        
        //Queue the asynchronous send requests
        for (int rank = 0; rank < group->Size(); rank++)
        {
            // Note that self-to-self transactions are handled using memcpy() in the Unpack() function
            if ((rank != group->Rank()) && (receiveSizes[rank] > 0))
            {
                group->QueueReceive(receiveBuffer[rank], receiveSizes[rank], parallelChar, rank, &requestHandles[rank]);
            }
        }
        
        int numSendsTotal = 0;
        for (int rank = 0; rank < group->Size(); rank++)
        {
            //Send the data
            if ((rank != group->Rank()) && (sendSizes[rank] > 0))
            {
                group->BlockingSynchronousSend(sendBuffer[rank], sendSizes[rank], parallelChar, rank);
                numSendsTotal++;
            }
        }
        // resize the status array if need be
        if (numSendsTotal!=statusHandles.size()) statusHandles.resize(numSendsTotal);
        
        int listSize = requestHandles.size();
        group->AwaitAllAsynchronousOperations(listSize, requestHandles.data(), statusHandles.data());
        
        //Distribute the data
        Unpack();
    }
    
    //Considering using a predicate here...
    void DataExchangePattern::ResizeOutBuffer(int rank)
    {
        size_t totalSize = 0;
        for (const auto tr:transactions)
        {
            if (tr->Sender() == group->Rank()) totalSize += tr->GetPackedSize();
        }
        // Consider a wrapper for realloc() if downsizing, it is a lot faster!
        if (sendBufferIsAllocated[rank]) Cmf_Free(sendBuffer[rank]);
        sendBuffer[rank] = (char*)Cmf_Alloc(totalSize);
    }
    
    void DataExchangePattern::ResizeInBuffer(int rank)
    {
        size_t totalSize = 0;
        for (const auto tr:transactions)
        {
            if (tr->Receiver() == group->Rank()) totalSize += tr->GetPackedSize();
        }
        // Consider a wrapper for realloc() if downsizing, it is a lot faster!
        if (receiveBufferIsAllocated[rank]) Cmf_Free(receiveBuffer[rank]);
        receiveBuffer[rank] = (char*)Cmf_Alloc(totalSize);
    }
}