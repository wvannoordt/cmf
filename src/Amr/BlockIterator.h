#ifndef BLOCK_ITERATOR_H
#define BLOCK_ITERATOR_H
#include <vector>
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
    };
}

#endif
