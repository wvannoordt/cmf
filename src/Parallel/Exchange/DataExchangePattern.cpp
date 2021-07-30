#include "DataExchangePattern.h"
#include "CmfScreen.h"
#include "CmfGC.h"
#include <algorithm>
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
    }
    
    void DataExchangePattern::SortByPriority(void)
    {
        std::vector<IDataTransaction*>& transactions = this->items;
        auto sortRule = [](IDataTransaction* const& a, IDataTransaction* const& b) -> bool { return (a->Priority() > b->Priority()); };
        std::sort(transactions.begin(), transactions.end(), sortRule);
    }
    
    void DataExchangePattern::OnAdd(IDataTransaction* transaction)
    {
        ComputeDevice sender = transaction->Sender();
        ComputeDevice receiver = transaction->Receiver();
        ComputeDevice currentRank = group->Rank();
        if ((sender == currentRank) || (receiver == currentRank))
        {
            if ((sender == currentRank) && (transaction->GetPackedSize() > 0))
            {
                resizeOutBufferRequired[receiver.id] = true;
                sendSizes[receiver.id] += transaction->GetPackedSize();
            }
            if ((receiver == currentRank) && (transaction->GetPackedSize() > 0))
            {
                resizeInBufferRequired[sender.id] = true;
                receiveSizes[sender.id] += transaction->GetPackedSize();
            }
        }
    }
    
    void DataExchangePattern::Pack(void)
    {
        std::vector<IDataTransaction*>& transactions = this->items;
        pointerIndices = sendBuffer;
        for (const auto tr:transactions)
        {
            ComputeDevice currentRank = group->Rank();
            if (tr->Sender() == currentRank)
            {
                tr->Pack(pointerIndices[tr->Receiver().id]);
                pointerIndices[tr->Receiver().id] += tr->GetPackedSize();
            }
        }
    }
    
    void DataExchangePattern::Unpack(void)
    {
        pointerIndices = receiveBuffer;
        
        //Note that self-to-self transactions are not copied between
        //send and receive buffers on the same rank. Why would they be? :-)
        pointerIndices[group->Rank().id] = sendBuffer[group->Rank().id];
        std::vector<IDataTransaction*>& transactions = this->items;
        for (const auto tr:transactions)
        {
            ComputeDevice currentRank = group->Rank();
            if (tr->Receiver() == currentRank)
            {
                tr->Unpack(pointerIndices[tr->Sender().id]);
                pointerIndices[tr->Sender().id] += tr->GetPackedSize();
            }
        }
    }
    
    void DataExchangePattern::ExchangeData(void)
    {
        group->Synchronize();
        for (int rank = 0; rank < group->Size(); rank++)
        {
            if (resizeOutBufferRequired[rank] || !sendBufferIsAllocated[rank])    this->ResizeOutBuffer(rank);
            if (resizeInBufferRequired[rank]  || !receiveBufferIsAllocated[rank]) this->ResizeInBuffer(rank);
        }
        
        //Aggregate the data
        Pack();
        
        //Count the number of send requests
        int numReceivesTotal = 0;
        for (int rank = 0; rank < group->Size(); rank++)
        {
            if ((rank != group->Rank().id) && (receiveSizes[rank] > 0))
            {
                numReceivesTotal++;
            }
        }
        
        // resize the status array if need be
        if (numReceivesTotal!=statusHandles.size())  statusHandles.resize(numReceivesTotal);
        if (numReceivesTotal!=requestHandles.size()) requestHandles.resize(numReceivesTotal);
        
        //Queue the asynchronous send requests
        int counter = 0;
        for (int rank = 0; rank < group->Size(); rank++)
        {
            // Note that self-to-self transactions are handled using memcpy() in the Unpack() function
            if ((rank != group->Rank().id) && (receiveSizes[rank] > 0))
            {
                group->QueueReceive(receiveBuffer[rank], receiveSizes[rank], parallelChar, rank, &requestHandles[counter]);
                counter++;
            }
        }   
        
        for (int rank = 0; rank < group->Size(); rank++)
        {
            //Send the data
            if ((rank != group->Rank().id) && (sendSizes[rank] > 0))
            {
                group->BlockingSynchronousSend(sendBuffer[rank], sendSizes[rank], parallelChar, rank);
            }
        }
        
        group->AwaitAllAsynchronousOperations(numReceivesTotal, requestHandles.data(), statusHandles.data());
        
        //Distribute the data
        Unpack();
        group->Synchronize();
    }
    
    //Considering using a predicate here...
    void DataExchangePattern::ResizeOutBuffer(int rank)
    {
        size_t totalSize = 0;
        std::vector<IDataTransaction*>& transactions = this->items;
        for (const auto tr:transactions)
        {
            if (tr->Sender() == group->Rank()) totalSize += tr->GetPackedSize();
        }
        // Consider a wrapper for realloc() if downsizing, it is a lot faster!
        if (sendBufferIsAllocated[rank]) Cmf_Free(sendBuffer[rank]);
        sendBuffer[rank] = (char*)Cmf_Alloc(totalSize);
        resizeOutBufferRequired[rank] = false;
        sendBufferIsAllocated[rank] = true;
    }
    
    void DataExchangePattern::ResizeInBuffer(int rank)
    {
        size_t totalSize = 0;
        std::vector<IDataTransaction*>& transactions = this->items;
        for (const auto tr:transactions)
        {
            if (tr->Receiver() == group->Rank()) totalSize += tr->GetPackedSize();
        }
        // Consider a wrapper for realloc() if downsizing, it is a lot faster!
        if (receiveBufferIsAllocated[rank]) Cmf_Free(receiveBuffer[rank]);
        receiveBuffer[rank] = (char*)Cmf_Alloc(totalSize);
        resizeInBufferRequired[rank] = false;
        receiveBufferIsAllocated[rank] = true;
    }
}