#include "NodeIterator.h"
namespace Anaptric
{
    NodeIterator::NodeIterator(void)
    {
        idx = 0;
    }
    
    NodeIterator::~NodeIterator(void){}
    
    void NodeIterator::Reset(void){idx = 0;}
    bool NodeIterator::IsAtEnd(void){return (idx==terminalNodes.size());}
    void NodeIterator::Increment(void){idx++;}
}