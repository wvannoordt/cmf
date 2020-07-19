#ifndef RefinementBlock_H
#define RefinementBlock_H

#include <string>
#include "PropTreeLib.h"

namespace gTree
{
    class RefinementBlock
    {
        public:
            RefinementBlock(std::string title);
            ~RefinementBlock(void);
            void Print(void);
        private:
            PropTreeLib::PropertyTree localInput;
            int* blockDim;
            int blockElemSize;
            double* blockBounds;

    };
}

#endif
