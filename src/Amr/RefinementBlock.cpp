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
        DefineTrunks();
    }

    RefinementBlock::~RefinementBlock(void)
    {
        this->Destroy();
    }

    void RefinementBlock::DefineTrunks(void)
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
            trunks[i] = new RefinementTreeNode(localBounds, 0, 0, 0, NULL);
        }
        for (int i = 0; i < totalNumTrunks; i++)
        {            
            int blockIndex[DIM];
            Dim2Idx(i, blockDim, blockIndex);
            int totalNumNeighbors = IS3D?27:9;
            int boxdim[DIM];
            __dloop(boxdim[d] = 3);
            int neighborBlockIndex[DIM];
            __dloop(neighborBlockIndex[d] = 0);
            for (int k = 0; k < totalNumNeighbors; k++)
            {
                int deltaijk[DIM];
                Dim2Idx(k, boxdim, deltaijk);
                __dloop(deltaijk[d] -= 1);
                __dloop(neighborBlockIndex[d] = blockIndex[d]+deltaijk[d]);
                bool targetIsSelf = true;
                __dloop(targetIsSelf = targetIsSelf&&(deltaijk[d]==0));
                if (!targetIsSelf)
                {
                    char wasPeriodic = 0;
                    int newNeighborIndex = Idx2DimPeriodic(blockDim, neighborBlockIndex, &wasPeriodic);
                    trunks[i]->CreateNewNeighbor(trunks[newNeighborIndex], deltaijk, wasPeriodic);
                }
            }
        }
    }
    
    void RefinementBlock::RefineAt(double coords[DIM], char refinementType)
    {
        RefinementTreeNode* target = GetNodeAt(coords);
        if (target)
        {
            target->Refine(refinementType);
        }
    }
    
    RefinementTreeNode* RefinementBlock::GetNodeAt(double coords[DIM])
    {
        int idx[DIM];
        if (!PointIsInDomain(coords, idx)) return trunks[Idx2Dim(blockDim, idx)]->RecursiveGetNodeAt(coords);
        else
        {
            HandleRefinementQueryOutsideDomain(coords);
            return NULL;
        }
    }
    
    bool RefinementBlock::PointIsInDomain(double coords[DIM], int* idx)
    {
        bool queryOutsideDomain = false;
        for (int d = 0; d < DIM; d++) 
        {
            idx[d] = (coords[d] - blockBounds[2*d])/(dx[d]);
            queryOutsideDomain = queryOutsideDomain  || (coords[d]<blockBounds[2*d]) || (coords[d]>=blockBounds[2*d+1]);
        }
    }
    
    bool RefinementBlock::PointIsInDomain(double coords[DIM])
    {
        int idx[DIM];
        return PointIsInDomain(coords, idx);
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

    void RefinementBlock::Render(TikzObject* picture)
    {
        picture->PushLineType(TikzLineType::solid, 0.1, TikzColor::black);
        for (int i = 0; i < totalNumTrunks; i++)
        {
            trunks[i]->DrawToObject(picture);
        }
        picture->PopLineType();
    }
}
