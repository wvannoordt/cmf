#ifndef NEIGH_ITER_H
#define NEIGH_ITER_H

#include "RefinementTreeNode.h"

namespace cmf
{
    /// @brief A class that is used to iterate over the neighbors of a node.
    /// @author WVN
    class NeighborIterator
    {
        public:
            /// @brief Constructor for the iterator
            /// @param host_in The node to iterate over the neighbors of
            /// @author WVN
            NeighborIterator(RefinementTreeNode* host_in);
            
            /// @brief Determines if the iterator has reached the end of the list.
            /// @author WVN
            bool Active(void);
            
            /// @brief Used to increment the iterator
            /// @param r Unused.
            /// @author WVN
            void operator++(int r);
            
            /// @brief Returns the current neighbor
            /// @author WVN
            RefinementTreeNode* Node(void);
            
            /// @brief Returns the current neighbor edge information
            /// @author WVN
            NodeEdge Edge(void);
        private:
            /// @brief The node whose neighbors are being iterated over
            RefinementTreeNode* host;
            
            /// @brief An underlying iterator object
            std::vector<std::pair<RefinementTreeNode*, NodeEdge>>::iterator iter;
    };
}

#endif
