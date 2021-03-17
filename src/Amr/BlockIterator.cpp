#include "BlockIterator.h"
#include "CmfError.h"
#include "RefinementTreeNode.h"
#include "CmfPrint.h"
#include "ParallelGroup.h"
namespace cmf
{
    BlockIterator::BlockIterator(IBlockIterable* hostBlock_in)
    {
        Build(hostBlock_in, BlockFilters::Terminal, IterableMode::serial);
    }
    
    BlockIterator::BlockIterator(IBlockIterable* hostBlock_in, NodeFilter_t filter_in)
    {
        Build(hostBlock_in, filter_in, IterableMode::serial);
    }
    
    BlockIterator::BlockIterator(IBlockIterable* hostBlock_in, IterableMode::IterableMode mode_in)
    {
        Build(hostBlock_in, BlockFilters::Terminal, mode_in);
    }
    
    BlockIterator::BlockIterator(IBlockIterable* hostBlock_in, NodeFilter_t filter_in, IterableMode::IterableMode mode_in)
    {
        Build(hostBlock_in, filter_in, mode_in);
    }
    
    void BlockIterator::SeekFirst(void)
    {
        if ((parallelMode == IterableMode::parallel) && CMF_PARALLEL)
        {
            if (!(hostBlock->ParallelPartitionContainsNode((*allNodes)[index])&&filter((*allNodes)[index]))){(*this)++;}
        }
        else
        {
            if (!(filter((*allNodes)[index]))){(*this)++;}
        }
        
    }
    
    size_t BlockIterator::Size(void)
    {
        return hostBlock->Size();
    }
    
    void BlockIterator::Build(IBlockIterable* hostBlock_in, NodeFilter_t filter_in, IterableMode::IterableMode mode_in)
    {
        hostBlock = hostBlock_in;
        allNodes = hostBlock_in->GetAllNodes();
        index = 0;
        filter = filter_in;
        parallelMode = mode_in;
        isAtEnd = false;
        SeekFirst();
    }

    BlockIterator::~BlockIterator(void)
    {

    }

    BlockIterator BlockIterator::operator++(int dummy)
    {
        index++;
        isAtEnd = index>=allNodes->size();
        if ((parallelMode == IterableMode::parallel) && CMF_PARALLEL)
        {
            while (!isAtEnd && !(hostBlock->ParallelPartitionContainsNode((*allNodes)[index])&&filter((*allNodes)[index])))
            {
                index++;
                isAtEnd = (index>=allNodes->size());
            }
        }
        else
        {
            while (!isAtEnd && !(filter((*allNodes)[index])))
            {
                index++;
                isAtEnd = (index>=allNodes->size());
            }
        }
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
