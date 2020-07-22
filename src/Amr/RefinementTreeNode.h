#ifndef RefinementTreeNode_H
#define RefinementTreeNode_H

#include <string>
#include "PropTreeLib.h"

namespace gTree
{
    class RefinementTreeNode
    {
        public:
            RefinementTreeNode(void);
            ~RefinementTreeNode(void);
            void Destroy(void);
        private:
            double* blockBounds;
            char refineType;

    };
}

#endif
