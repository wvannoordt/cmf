#include "IDataTransaction.h"

namespace cmf
{
    IDataTransaction::IDataTransaction(ComputeDevice sender_in, ComputeDevice receiver_in)
    {
        sender = sender_in;
        receiver = receiver_in;
    }
    
    ComputeDevice IDataTransaction::Sender(void)
    {
        return sender;
    }
    
    ComputeDevice IDataTransaction::Receiver(void)
    {
        return receiver;
    }
}