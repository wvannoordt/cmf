#ifndef RefinementBlock_H
#define RefinementBlock_H

#include <string>
#include "PropTreeLib.h"
#include "RefinementTreeNode.h"

namespace gTree
{
    class RefinementBlock
    {
        public:
            RefinementBlock(std::string title);
            ~RefinementBlock(void);
            void Print(void);
            void Destroy(void);
            void Render(std::string filename);
        private:
            void Allocate(void);
            PropTreeLib::PropertyTree localInput;
            int* blockDim;
            int totalNumTrunks;
            double* blockBounds;
            RefinementTreeNode** trunks;
            bool deallocTrunks;

    };
}

#endif
