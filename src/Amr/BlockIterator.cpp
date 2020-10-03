#include "BlockIterator.h"

namespace cmf
{
    BlockIterator::BlockIterator(RefinementBlock* hostBlock_in)
    {
        hostBlock = hostBlock_in;
        index = 0;
    }
    
    BlockIterator::~BlockIterator(void)
    {
        
    }
    
    BlockIterator& BlockIterator::operator++(int dummy)
    {
        index++;
        return *this;
    }
    
    bool BlockIterator::HasNext(void)
    {
        return (index<10);
    }
}
