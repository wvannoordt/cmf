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
            ~BlockIterator(void);
            bool HasNext(void);
            BlockIterator   operator++(int dummy);
            BlockIterator & operator++(void);
            friend std::ostream & operator << (std::ostream &out, const BlockIterator &c) {out << c.index; return out;}
            friend CmfOutputStream & operator << (CmfOutputStream &out, const BlockIterator &c) {out << c.index; return out;}
        private:
            RefinementBlock* hostBlock;
            size_t index;
            NodeFilter_t filter;
    };
}

#endif
