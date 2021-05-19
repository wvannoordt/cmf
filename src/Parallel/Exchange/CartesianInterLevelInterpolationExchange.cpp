#include "CartesianInterLevelInterpolationExchange.h"

namespace cmf
{
    CartesianInterLevelInterpolationExchange::CartesianInterLevelInterpolationExchange(int sendRank_in, int recvRank_in)
        : IDataTransaction(sendRank_in, recvRank_in)
    {
        
    }
    
    size_t CartesianInterLevelInterpolationExchange::GetPackedSize(void)
    {
        return 0;
    }
    
    void CartesianInterLevelInterpolationExchange::Pack(char* buf)
    {
        //todo
    }
    
    void CartesianInterLevelInterpolationExchange::Unpack(char* buf)
    {
        //todo
    }
}