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
        filter = BlockFilters::Terminal;
        isAtEnd = false;
    }
    
    BlockIterator::BlockIterator(RefinementBlock* hostBlock_in, NodeFilter_t filter_in)
    {
        hostBlock = hostBlock_in;
        allNodes = &(hostBlock_in->allNodes);
        index = 0;
        filter = filter_in;
        isAtEnd = false;
        SeekFirst();
    }
    
    void BlockIterator::SeekFirst(void)
    {
        if (!filter((*allNodes)[index])){(*this)++;}
    }
    
    size_t BlockIterator::Size(void)
    {
        return hostBlock->Size();
    }

    BlockIterator::~BlockIterator(void)
    {

    }

    BlockIterator BlockIterator::operator++(int dummy)
    {
        index++;
        isAtEnd = index>=allNodes->size();
        while (!isAtEnd && !(filter((*allNodes)[index]))){index++;isAtEnd = (index>=allNodes->size());}
        return *this;
    }

    BlockIterator & BlockIterator::operator++(void)
    {
        index++;
        isAtEnd = index>=allNodes->size();
        while (!isAtEnd && !(filter((*allNodes)[index]))){index++; isAtEnd = (index>=allNodes->size());}
        return *this;
    }

    RefinementTreeNode* BlockIterator::Node(void)
    {
        return (*allNodes)[index];
    }

    bool BlockIterator::HasNext(void)
    {
        return !isAtEnd;
    }
}
