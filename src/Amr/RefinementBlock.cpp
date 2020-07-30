#include <string>
#include <iostream>
#include "PropTreeLib.h"
#include "gTree.h"
#include "RefinementBlock.h"
#include "RefinementTreeNode.h"
#include "Config.h"
#include "TikzObject.h"
#include "Utils.hx"
#include "RefinementConstraint.h"

namespace gTree
{
    RefinementBlock::RefinementBlock(std::string title)
    {
        srand((unsigned int)time(NULL));
        localInput.SetAsSubtree(mainInput[title]);
        localInput["blockDim"].MapTo(&blockDim) = new PropTreeLib::Variables::PTLStaticIntegerArray(DIM, "Base block dimensions");
        localInput["blockBounds"].MapTo(&blockBounds) = new PropTreeLib::Variables::PTLStaticDoubleArray(2*DIM, "Base block bounds");
        localInput["refinementConstraintType"].MapTo((int*)&refinementConstraintType) = new PropTreeLib::Variables::PTLAutoEnum(RefinementConstraint::free, RefinementConstraintStr, "Determines how refinements are constrained");
        localInput.StrictParse();
        totalNumTrunks = 1;
        for (int i = 0; i < DIM; i++) totalNumTrunks*=blockDim[i];
        for (int d = 0; d < DIM; d++) dx[d] = (blockBounds[2*d+1]-blockBounds[2*d])/blockDim[d];
        Allocate();
    }

    RefinementBlock::~RefinementBlock(void)
    {
        this->Destroy();
    }

    void RefinementBlock::Allocate(void)
    {
        deallocTrunks = true;
        trunks = new RefinementTreeNode* [totalNumTrunks];
        double localBounds[2*DIM];
        int idx[DIM];
        for (int i = 0; i < totalNumTrunks; i++)
        {
            Dim2Idx(i, blockDim, idx);
            for (int d = 0; d < DIM; d++)
            {
                localBounds[2*d] = blockBounds[2*d]+idx[d]*dx[d];
                localBounds[2*d+1] = blockBounds[2*d]+(idx[d]+1)*dx[d];
            }
            trunks[i] = new RefinementTreeNode(localBounds, 0, 0, 0);
        }
    }
    
    void RefinementBlock::RefineAt(double coords[DIM], char refinementType)
    {
        int idx[DIM];
        bool queryOutsideDomain = false;
        for (int d = 0; d < DIM; d++) 
        {
            idx[d] = (coords[d] - blockBounds[2*d])/(dx[d]);
            queryOutsideDomain = queryOutsideDomain  || (coords[d]<blockBounds[2*d]) || (coords[d]>=blockBounds[2*d+1]);
        }
        if (!queryOutsideDomain) trunks[Idx2Dim(blockDim, idx)]->RecursiveRefineAt(coords, refinementType);
        else HandleRefinementQueryOutsideDomain(coords);
    }
    
    void RefinementBlock::HandleRefinementQueryOutsideDomain(double coords[DIM])
    {
        //Extend domain? crash? ignore?
    }

    void RefinementBlock::RefineAll(char refinementType)
    {
        for (int i = 0; i < totalNumTrunks; i++)
        {
            trunks[i]->Refine(refinementType);
        }
    }

    void RefinementBlock::RefineRandom()
    {
        for (int i = 0; i < totalNumTrunks; i++)
        {
            if (RandomInt(5)<4)
            {
                trunks[i]->RefineRandom();
            }
        }
    }

    void RefinementBlock::Destroy(void)
    {
        if (deallocTrunks)
        {
            for (int i = 0; i < totalNumTrunks; i++)
            {
                trunks[i]->Destroy();
                delete trunks[i];
            }
            delete[] trunks;
        }
    }

    void RefinementBlock::Print(void)
    {
        localInput.DebugPrint();
    }

    void RefinementBlock::Render(std::string filename)
    {
        TikzObject picture;
        picture.PushLineType(TikzLineType::solid, 0.2, TikzColor::black);
        picture.Open(filename);
        for (int i = 0; i < totalNumTrunks; i++)
        {
            trunks[i]->DrawToObject(&picture);
        }
        picture.Close();
    }
}
