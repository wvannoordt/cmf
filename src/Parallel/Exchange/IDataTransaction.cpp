#include "IDataTransaction.h"

namespace cmf
{
    IDataTransaction::IDataTransaction(int sender_in, int receiver_in)
    {
        sender = sender_in;
        receiver = receiver_in;
    }
    IDataTransaction::~IDataTransaction(void)
    {
        
    }
    
    int IDataTransaction::Sender(void)
    {
        return sender;
    }
    
    int IDataTransaction::Receiver(void)
    {
        return receiver;
    }
}