#include "DataExchangePattern.h"
#include "CmfScreen.h"
#include "CmfPrint.h"
#include "CmfGC.h"
#include <algorithm>
#include <unistd.h>
namespace cmf
{
    DataExchangePattern::DataExchangePattern(ParallelGroup* group_in)
    {
        group = group_in;
        int groupSize = group->Size();
        
        //GPU stuff
        int numDevices = group->GetCudaDevices()->NumDevices();
        gpus.resize(numDevices);
        for (int i = 0; i < numDevices; i++)
        {
            gpus[i].isGpu = true;
            gpus[i].id = group->Rank().id;
            gpus[i].deviceNum = i;
        }
        
        int totalNumBuffers = groupSize + numDevices;
        resizeOutBufferRequired.resize(totalNumBuffers, true);
        resizeInBufferRequired.resize(totalNumBuffers, true);
        sendBufferIsAllocated.resize(totalNumBuffers, false);
        receiveBufferIsAllocated.resize(totalNumBuffers, false);
        sendBuffer.resize(totalNumBuffers, NULL);
        receiveBuffer.resize(totalNumBuffers, NULL);
        requestHandles.resize(totalNumBuffers);
        sendSizes.resize(totalNumBuffers, 0);
        receiveSizes.resize(totalNumBuffers, 0);
    }
    
    DataExchangePattern::~DataExchangePattern(void)
    {
        for (int rank = 0; rank < sendBuffer.size(); rank++)
        {
            if (sendBufferIsAllocated[rank])
            {
                sendBufferIsAllocated[rank] = false;
                if (rank>=group->Size()) { Cmf_GpuFree(sendBuffer[rank]); }
                else { Cmf_Free(sendBuffer[rank]); }
            }
            if (receiveBufferIsAllocated[rank])
            {
                receiveBufferIsAllocated[rank] = false;
                if (rank>=group->Size()) { Cmf_GpuFree(receiveBuffer[rank]); }
                else { Cmf_Free(receiveBuffer[rank]); }
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
        
        if (sender.id   == currentRank.id) resizeOutBufferRequired[receiver.id] = true;
        if (receiver.id == currentRank.id) resizeInBufferRequired [sender.id]   = true;
        if (sender.id   == currentRank.id && sender.isGpu)   resizeOutBufferRequired[group->Size()+sender.deviceNum] = true;
        if (receiver.id == currentRank.id && receiver.isGpu) resizeInBufferRequired[group->Size()+sender.deviceNum] = true;
        
    }
    
    void DataExchangePattern::Pack(void)
    {
        std::vector<IDataTransaction*>& transactions = this->items;
        pointerIndices = sendBuffer;
        for (const auto tr:transactions)
        {
            ComputeDevice currentRank = group->Rank(); //This call will never return a GPU device
            if (tr->Sender().id == currentRank.id)
            {
                int sendIndex = tr->Sender().id;
                int recvIndex = tr->Receiver().id;
                if (tr->Sender().isGpu)   sendIndex += group->Size();
                if (tr->Receiver().isGpu) recvIndex += group->Size();
                tr->Pack(pointerIndices[recvIndex]);
                pointerIndices[recvIndex] += tr->GetPackedSize();
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
            ComputeDevice currentRank = group->Rank(); //This call will never return a GPU device
            if (tr->Receiver().id == currentRank.id)
            {
                int sendIndex = tr->Sender().id;
                int recvIndex = tr->Receiver().id;
                if (tr->Sender().isGpu)   sendIndex += group->Size();
                if (tr->Receiver().isGpu) recvIndex += group->Size();
                tr->Unpack(pointerIndices[sendIndex]);
                pointerIndices[sendIndex] += tr->GetPackedSize();
            }
        }
    }
    
    void DataExchangePattern::ExchangeData(void)
    {
        group->Synchronize();
        for (int rank = 0; rank < sendBuffer.size(); rank++)
        {
            if (resizeOutBufferRequired[rank] || !sendBufferIsAllocated[rank])    this->ResizeOutBuffer(rank);
            if (resizeInBufferRequired[rank]  || !receiveBufferIsAllocated[rank]) this->ResizeInBuffer(rank);
        }
        
        //Aggregate the data
        Pack();
        
        int rankMin = group->Size();
        int rankMax = sendBuffer.size();
        for (int i = rankMin; i < rankMax; i++)
        {
            GpuMemTransfer<DeviceTransferDirection::GpuToGpu>((void*)sendBuffer[i], (void*)receiveBuffer[i], sendSizes[i], i-group->Size());
        }
        
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
        CMF_CUDA_CHECK(cudaDeviceSynchronize());
        
        //Distribute the data
        Unpack();
        group->Synchronize();
        
    }
    
    //Considering using a predicate here...
    void DataExchangePattern::ResizeOutBuffer(int rank)
    {
        size_t totalSize = 0;
        std::vector<IDataTransaction*>& transactions = this->items;
        
        bool rankIsGpu = false;
        ComputeDevice device;
        if (rank >= group->Size())
        {
            device = gpus[rank-group->Size()];
            rankIsGpu = true;
        }
        
        for (const auto tr:transactions)
        {
            if ((tr->Sender().id == group->Rank().id) && (tr->Receiver().id==rank))
            {
                if (rankIsGpu==tr->Sender().isGpu) totalSize += tr->GetPackedSize();
            }
        }

        // Consider a wrapper for realloc() if downsizing, it is a lot faster!
        if (rankIsGpu)
        {
            if (sendBufferIsAllocated[rank]) Cmf_GpuFree(sendBuffer[rank]);
            sendBuffer[rank] = (char*)Cmf_GpuAlloc(totalSize, device.deviceNum);
            sendSizes[rank] = totalSize;
        }
        else
        {
            if (sendBufferIsAllocated[rank]) Cmf_Free(sendBuffer[rank]);
            sendBuffer[rank] = (char*)Cmf_Alloc(totalSize);
            sendSizes[rank] = totalSize;
        }
        resizeOutBufferRequired[rank] = false;
        sendBufferIsAllocated[rank] = true;
    }
    
    void DataExchangePattern::ResizeInBuffer(int rank)
    {
        size_t totalSize = 0;
        std::vector<IDataTransaction*>& transactions = this->items;
        
        bool rankIsGpu = false;
        ComputeDevice device;
        if (rank >= group->Size())
        {
            device = gpus[rank-group->Size()];
            rankIsGpu = true;
        }
        
        for (const auto tr:transactions)
        {
            if ((tr->Receiver().id == group->Rank().id) && (tr->Sender().id==rank))
            {
                if (rankIsGpu == tr->Receiver().isGpu) totalSize += tr->GetPackedSize();
            }
        }
        
        
        // Consider a wrapper for realloc() if downsizing, it is a lot faster!
        if (rankIsGpu)
        {
            if (receiveBufferIsAllocated[rank]) Cmf_GpuFree(receiveBuffer[rank]);
            receiveBuffer[rank] = (char*)Cmf_GpuAlloc(totalSize, device.deviceNum);
            receiveSizes[rank] = totalSize;
        }
        else
        {
            if (receiveBufferIsAllocated[rank]) Cmf_Free(receiveBuffer[rank]);
            receiveBuffer[rank] = (char*)Cmf_Alloc(totalSize);
            receiveSizes[rank] = totalSize;
        }
        resizeInBufferRequired[rank] = false;
        receiveBufferIsAllocated[rank] = true;
    }
    
    void DataExchangePattern::ForceResizeBuffers(void)
    {
        for (int rank = 0; rank < sendBuffer.size(); rank++)
        {
            this->ResizeInBuffer (rank);
            this->ResizeOutBuffer(rank);
        }
    }
    
    void DataExchangePattern::DebugPrint(void)
    {
        if (group->IsRoot()) print("Data exchange pattern over parallel group of size", group->Size());
        for (int p = 0; p < group->Size(); p++)
        {
            if (p == group->Rank().id)
            {
                print("---------------------------------");
                print("Rank:", group->Rank());
                print("Group size:", group->Size());
                print("Send buffers:");
                for (int i = 0; i < sendBuffer.size(); i++)
                {
                    if (i==group->Size()) print("*");
                    print(i, sendSizes[i]);
                }
                print("Recv buffers:");
                for (int i = 0; i < receiveBuffer.size(); i++)
                {
                    if (i==group->Size()) print("*");
                    print(i, receiveSizes[i]);
                }
                print("Transactions:");
                std::vector<IDataTransaction*>& transactions = this->items;
                for (auto tr: transactions)
                {
                    if (tr->Sender().id==1 && tr->Receiver().id==0)
                    {
                        print(tr->Sender().id, "->", tr->Receiver().id, tr->GetPackedSize());
                    }
                }
                print("---------------------------------");
            }
            usleep(10000);
            group->Synchronize();
        }
    }
}