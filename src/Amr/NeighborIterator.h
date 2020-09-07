#ifndef NEIGH_ITER_H
#define NEIGH_ITER_H

#include "RefinementTreeNode.h"

namespace Anaptric
{
    class NeighborIterator
    {
        public:
            NeighborIterator(RefinementTreeNode* host_in);
            bool Active(void);
            void operator++(int r);
            RefinementTreeNode* Node(void);
            NodeEdge Edge(void);
        private:
            RefinementTreeNode* host;
            std::map<RefinementTreeNode*, NodeEdge>::iterator iter;
    };
}

#endif