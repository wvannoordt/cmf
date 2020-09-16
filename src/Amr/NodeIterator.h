#ifndef NodeIterator_H
#define NodeIterator_H

class RefinementTreeNode;
#include "RefinementTreeNode.h"

namespace Anaptric
{
    class NodeIterator
    {
        public:
            NodeIterator(void);
            ~NodeIterator(void);
            void Reset(void);
            bool IsAtEnd(void);
            void Increment(void);
        private:
            int idx;
            std::vector<RefinementTreeNode*> terminalNodes;
    };
}

#endif