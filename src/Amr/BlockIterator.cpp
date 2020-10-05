#include "BlockIterator.h"
#include "CmfError.h"
#include "RefinementTreeNode.h"
namespace cmf
{
    BlockIterator::BlockIterator(RefinementBlock* hostBlock_in)
    {
        hostBlock = hostBlock_in;
        index = 0;
        filter = BlockFilters::Every;
    }

    BlockIterator::~BlockIterator(void)
    {

    }

    BlockIterator BlockIterator::operator++(int dummy)
    {
        index++;
        return *this;
    }
    
    BlockIterator & BlockIterator::operator++(void)
    {
        index++;
        return *this;
    }
    
    RefinementTreeNode* BlockIterator

    bool BlockIterator::HasNext(void)
    {
        return (index<10);
    }
}
