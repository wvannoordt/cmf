#ifndef RefinementTreeNode_H
#define RefinementTreeNode_H

#include <string>
#include "Config.h"
#include "TikzObject.h"

namespace gTree
{
    class RefinementTreeNode
    {
        public:
            RefinementTreeNode(double* hostBounds, char refineType_in, char refineOrientation_in);
            ~RefinementTreeNode(void);
            void Destroy(void);
            void Refine(char newRefinementType);
            void RefineRandom();
            void DrawToObject(TikzObject* picture);
        private:
            void RefineLocal(char newRefinementType);
            void DefineBounds(double* hostBounds, char refineType_in, char refineOrientation_in);
            char refineType, refineOrientation;
            bool isTerminal, deallocSubTrees;
            double blockBounds[2*DIM];
            RefinementTreeNode** subNodes;
            int numSubNodes;

    };
}

#endif
