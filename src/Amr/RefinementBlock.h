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
            void RefineAt(double coords[DIM], char refinementType);
            RefinementTreeNode* GetNodeAt(double coords[DIM]);
            bool PointIsInDomain(double coords[DIM], int* idx);
            bool PointIsInDomain(double coords[DIM]);
        private:
            void DefineTrunks(void);
            void HandleRefinementQueryOutsideDomain(double coords[DIM]);
            PropTreeLib::PropertyTree localInput;
            int* blockDim;
            int totalNumTrunks;
            double* blockBounds;
            double dx[DIM];
            RefinementTreeNode** trunks;
            bool deallocTrunks;
            RefinementConstraint::RefinementConstraint refinementConstraintType;

    };
}

#endif
