#include "BlockIterator.h"
#include "CmfError.h"
#include "RefinementTreeNode.h"
namespace cmf
{
    BlockIterator::BlockIterator(RefinementBlock* hostBlock_in)
    {
        hostBlock = hostBlock_in;
        allNodes = &(hostBlock_in->allNodes);
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

    RefinementTreeNode* BlockIterator::Node(void)
    {
        return (*allNodes)[index];
    }

    bool BlockIterator::HasNext(void)
    {
        return (index<allNodes->size());
    }
}
