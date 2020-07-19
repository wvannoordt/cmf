#ifndef REFINEMENTTREE_H
#define REFINEMENTTREE_H

#include <string>
#include "PropTreeLib.h"

namespace gTree
{
    class RefinementTree
    {
        public:
            RefinementTree(std::string title);
            ~RefinementTree(void);
        private:
            PropTreeLib::PropertyTree localInput;
            
    };
}

#endif
