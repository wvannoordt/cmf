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
            void Print(void);
            void Destroy(void);
        private:
            PropTreeLib::PropertyTree localInput;
            int* blockDim;
            int blockElemSize;
            double* blockBounds;

    };
}

#endif
