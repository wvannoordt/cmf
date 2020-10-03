#ifndef BLOCK_ITERATOR_H
#define BLOCK_ITERATOR_H
#include <vector>
#include "AmrFcnTypes.h"
namespace cmf
{
    class RefinementBlock;
    class BlockIterator
    {
        public:
            BlockIterator(RefinementBlock* hostBlock_in);
            ~BlockIterator(void);
            bool HasNext(void);
            BlockIterator& operator++(int dummy);
        private:
            RefinementBlock* hostBlock;
            size_t index;
            NodeFilter_t filter;
    };
}

#endif
