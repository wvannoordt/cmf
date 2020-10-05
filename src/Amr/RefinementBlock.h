#ifndef RefinementBlock_H
#define RefinementBlock_H

#include <string>
#include <vector>
#include "AmrFcnTypes.h"
#include "PropTreeLib.h"
#include "RefinementTreeNode.h"
#include "BlockIterator.h"
#include "RefinementConstraint.h"

namespace cmf
{
    class RefinementTreeNode;
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
            void RefineAt(double coords[CMF_DIM], char refinementType);
            RefinementTreeNode* GetNodeAt(double coords[CMF_DIM]);
            bool PointIsInDomain(double coords[CMF_DIM], int* idx);
            bool PointIsInDomain(double coords[CMF_DIM]);
            void SetRefineLimitCriterion(NodeFilter_t limiter_in);
            void OutputDebugVtk(std::string filename);
            void RegisterNewNode(RefinementTreeNode* newNode);
        private:
            void DefineTrunks(void);
            void HandleRefinementQueryOutsideDomain(double coords[CMF_DIM]);
            PropTreeLib::PropertyTree localInput;
            int* blockDim;
            int totalNumTrunks;
            double* blockBounds;
            double dx[CMF_DIM];
            RefinementTreeNode** trunks;
            bool deallocTrunks;
            RefinementConstraint::RefinementConstraint refinementConstraintType;
            NodeFilter_t refineLimiter;
            std::vector<RefinementTreeNode*> allNodes;
        friend class RefinementTreeNode;
    };
}

#endif
