#include "MultiTransaction.h"
#include "CmfError.h"
#include <cstring>
#include "CmfPrint.h"

namespace cmf
{
    MultiTransaction::MultiTransaction(
        void* sendTarget_in, std::vector<size_t>& sendOffsets_in, std::vector<size_t>& sendSizes_in, int sendRank_in,
        void* recvTarget_in, std::vector<size_t>& recvOffsets_in, std::vector<size_t>& recvSizes_in, int recvRank_in)
        : IDataTransaction(sendRank_in, recvRank_in)
    {
        sendTarget  = sendTarget_in;
        recvTarget  = recvTarget_in;
        
        sendOffsets = sendOffsets_in;
        recvOffsets = recvOffsets_in;
        sendSizes   = sendSizes_in;
        recvSizes   = recvSizes_in;
        
        if ((sendOffsets_in.size() != sendSizes_in.size()) ||(recvOffsets_in.size() != recvSizes_in.size()))
        {
            CmfError("A MultiTransaction has been created with inconsistent offsets and sizes");
        }
        size_t packedSizeSend = 0;
        for (const auto s:sendSizes) packedSizeSend += s;
        size_t packedSizeRecv = 0;
        for (const auto s:recvSizes) packedSizeRecv += s;
        if (packedSizeSend != packedSizeRecv)
        {
            CmfError("A MultiTransaction has been created with inconsistent total sizes");
        }
        packedSize = packedSizeRecv;
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
        char* cTarget = (char*)sendTarget;
        int numOffsets = sendOffsets.size();
        for (int i = 0; i < numOffsets; i++)
        {
            memcpy(copyTo, cTarget + sendOffsets[i], sendSizes[i]);
            copyTo += sendSizes[i];
        }
    }
    
    void MultiTransaction::Unpack(char* buf)
    {
        char* copyFrom = buf;
        char* cTarget = (char*)recvTarget;
        int numOffsets = recvOffsets.size();
        for (int i = 0; i < numOffsets; i++)
        {
            memcpy(cTarget + recvOffsets[i], copyFrom, recvSizes[i]);
            copyFrom += recvSizes[i];
        }
    }
}