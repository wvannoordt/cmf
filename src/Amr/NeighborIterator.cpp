#include "NeighborIterator.h"

namespace Anaptric
{
    NeighborIterator::NeighborIterator(RefinementTreeNode* host_in)
    {
        host = host_in;
        iter = host->neighbors.begin();
    }
    
    bool NeighborIterator::Active(void)
    {
        return iter!=host->neighbors.end();
    }
    
    RefinementTreeNode* NeighborIterator::Node(void)
    {
        return iter->first;
    }
    
    NodeEdge NeighborIterator::Edge(void)
    {
        return iter->second;
    }
    
    void NeighborIterator::operator++(int r)
    {
        iter++;
    }
}