#ifndef BLOCK_ITERATOR_H
#define BLOCK_ITERATOR_H
#include <vector>
#include <iostream>
#include <ostream>
#include "AmrFcnTypes.h"
#include "CmfOutputStream.h"
namespace cmf
{
    class RefinementBlock;
    class BlockIterator
    {
        public:
            BlockIterator(RefinementBlock* hostBlock_in);
            BlockIterator(RefinementBlock* hostBlock_in, NodeFilter_t filter_in);
            ~BlockIterator(void);
            bool HasNext(void);
            BlockIterator   operator++(int dummy);
            BlockIterator & operator++(void);
            RefinementTreeNode* Node(void);
            size_t Size(void);
            friend std::ostream & operator << (std::ostream &out, const BlockIterator &c) {out << c.index; return out;}
            friend CmfOutputStream & operator << (CmfOutputStream &out, const BlockIterator &c) {out << c.index; return out;}
        private:
            RefinementBlock* hostBlock;
            size_t index;
            NodeFilter_t filter;
            std::vector<RefinementTreeNode*>* allNodes;
    };
}

#endif
