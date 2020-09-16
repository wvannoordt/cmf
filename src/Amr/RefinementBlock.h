#ifndef RefinementBlock_H
#define RefinementBlock_H

#include <string>
#include "PropTreeLib.h"
#include "RefinementTreeNode.h"
#include "RefinementConstraint.h"

namespace Anaptric
{
    class RefinementBlock
    {
        public:
            RefinementBlock(std::string title);
            ~RefinementBlock(void);
            void Print(void);
            void Destroy(void);
            void Render(TikzObject* picture);
            void Render(TikzObject* picture, DebugTikzDraw_t debugger);
            void RefineAll(char refinementType);
            void RefineRandom();
            void RefineAt(double coords[ANA_DIM], char refinementType);
            RefinementTreeNode* GetNodeAt(double coords[ANA_DIM]);
            bool PointIsInDomain(double coords[ANA_DIM], int* idx);
            bool PointIsInDomain(double coords[ANA_DIM]);
            void SetRefineLimitCriterion(RefinementLimit_t limiter_in);
            void OutputDebugVtk(std::string filename);
        private:
            void DefineTrunks(void);
            void HandleRefinementQueryOutsideDomain(double coords[ANA_DIM]);
            PropTreeLib::PropertyTree localInput;
            int* blockDim;
            int totalNumTrunks;
            double* blockBounds;
            double dx[ANA_DIM];
            RefinementTreeNode** trunks;
            bool deallocTrunks;
            RefinementConstraint::RefinementConstraint refinementConstraintType;
            RefinementLimit_t refineLimiter;
            friend class NodeIterator;
    };
}

#endif
