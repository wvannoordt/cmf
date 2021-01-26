#include "MultiTransaction.h"

namespace cmf
{
    MultiTransaction::MultiTransaction(int sender_in, int receiver_in)
        : IDataTransaction(sender_in, receiver_in)
    {
        
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
        
    }
    
    void MultiTransaction::Unpack(char* buf)
    {
        
    }
}