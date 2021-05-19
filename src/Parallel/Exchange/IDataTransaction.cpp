#include "IDataTransaction.h"

namespace cmf
{
    IDataTransaction::IDataTransaction(int sender_in, int receiver_in)
    {
        sender = sender_in;
        receiver = receiver_in;
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