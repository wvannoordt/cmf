#ifndef RefinementBlock_H
#define RefinementBlock_H

#include <string>
#include "PropTreeLib.h"
#include "RefinementTreeNode.h"
#include "RefinementConstraint.h"

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
            void RefineAll(char refinementType);
            void RefineRandom();
        private:
            void Allocate(void);
            PropTreeLib::PropertyTree localInput;
            int* blockDim;
            int totalNumTrunks;
            double* blockBounds;
            RefinementTreeNode** trunks;
            bool deallocTrunks;
            RefinementConstraint::RefinementConstraint refinementConstraintType;

    };
}

#endif
