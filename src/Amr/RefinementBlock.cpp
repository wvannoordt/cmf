#include <string>
#include <iostream>
#include "PropTreeLib.h"
#include "Anaptric.h"
#include "RefinementBlock.h"
#include "RefinementTreeNode.h"
#include "Config.h"
#include "TikzObject.h"
#include "Utils.hx"
#include "RefinementConstraint.h"
#include "VtkFile.h"

namespace Anaptric
{
    RefinementBlock::RefinementBlock(std::string title)
    {
        refineLimiter = NULL;
        srand((unsigned int)time(NULL));
        localInput.SetAsSubtree(mainInput[title]);
        localInput["blockDim"].MapTo(&blockDim) = new PropTreeLib::Variables::PTLStaticIntegerArray(ANA_DIM, "Base block dimensions");
        localInput["blockBounds"].MapTo(&blockBounds) = new PropTreeLib::Variables::PTLStaticDoubleArray(2*ANA_DIM, "Base block bounds");
        localInput["refinementConstraintType"].MapTo((int*)&refinementConstraintType)
            = new PropTreeLib::Variables::PTLAutoEnum(RefinementConstraint::free, RefinementConstraintStr, "Determines how refinements are constrained");
        localInput.StrictParse();
        totalNumTrunks = 1;
        for (int i = 0; i < ANA_DIM; i++) totalNumTrunks*=blockDim[i];
        for (int d = 0; d < ANA_DIM; d++) dx[d] = (blockBounds[2*d+1]-blockBounds[2*d])/blockDim[d];
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
        double localBounds[2*ANA_DIM];
        int idx[ANA_DIM];
        for (int i = 0; i < totalNumTrunks; i++)
        {
            Dim2Idx(i, blockDim, idx);
            for (int d = 0; d < ANA_DIM; d++)
            {
                localBounds[2*d] = blockBounds[2*d]+idx[d]*dx[d];
                localBounds[2*d+1] = blockBounds[2*d]+(idx[d]+1)*dx[d];
            }
            trunks[i] = new RefinementTreeNode(localBounds, 0, 0, 0, NULL, refinementConstraintType);
            trunks[i]->SetRefineLimiter(&refineLimiter);
        }
        for (int i = 0; i < totalNumTrunks; i++)
        {
            int blockIndex[ANA_DIM];
            Dim2Idx(i, blockDim, blockIndex);
            int totalNumNeighbors = ANA_IS3D?27:9;
            int boxdim[ANA_DIM];
            __dloop(boxdim[d] = 3);
            int neighborBlockIndex[ANA_DIM];
            __dloop(neighborBlockIndex[d] = 0);
            for (int k = 0; k < totalNumNeighbors; k++)
            {
                int deltaijk[ANA_DIM];
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

    void RefinementBlock::SetRefineLimitCriterion(RefinementLimit_t limiter_in)
    {
        refineLimiter = limiter_in;
    }

    void RefinementBlock::RefineAt(double coords[ANA_DIM], char refinementType)
    {
        RefinementTreeNode* target = GetNodeAt(coords);
        if (target)
        {
            target->Refine(refinementType);
        }
    }

    RefinementTreeNode* RefinementBlock::GetNodeAt(double coords[ANA_DIM])
    {
        int idx[ANA_DIM];
        if (!PointIsInDomain(coords, idx)) return trunks[Idx2Dim(blockDim, idx)]->RecursiveGetNodeAt(coords);
        else
        {
            HandleRefinementQueryOutsideDomain(coords);
            return NULL;
        }
    }

    bool RefinementBlock::PointIsInDomain(double coords[ANA_DIM], int* idx)
    {
        bool queryOutsideDomain = false;
        for (int d = 0; d < ANA_DIM; d++)
        {
            idx[d] = (coords[d] - blockBounds[2*d])/(dx[d]);
            queryOutsideDomain = queryOutsideDomain  || (coords[d]<blockBounds[2*d]) || (coords[d]>=blockBounds[2*d+1]);
        }
    }

    bool RefinementBlock::PointIsInDomain(double coords[ANA_DIM])
    {
        int idx[ANA_DIM];
        return PointIsInDomain(coords, idx);
    }

    void RefinementBlock::HandleRefinementQueryOutsideDomain(double coords[ANA_DIM])
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

    //This is for debugging only. For any real VTK output, an externl iterator should be used.
    void RefinementBlock::OutputDebugVtk(std::string filename)
    {
        VtkFile output(filename, VtkFormatType::ascii, VtkTopologyType::unstructuredGrid);
        int totalNumBlocks = 0;
        for (int i = 0; i < totalNumTrunks; i++) trunks[i]->RecursiveCountTerminal(&totalNumBlocks);
        output.Mesh()->Component("DATASET")->SetAttribute("numPoints",   (ANA_IS3D?8:4)*totalNumBlocks);
        output.Mesh()->Component("DATASET")->SetAttribute("bufferCount", 3*(ANA_IS3D?8:4)*totalNumBlocks);
        output.Mesh()->Component("DATASET")->SetAttribute("stride", 3);
        output.Mesh()->Component("CELLS")->SetAttribute("numPoints", totalNumBlocks);
        output.Mesh()->Component("CELLS")->SetAttribute("bufferCount", (ANA_IS3D?9:5)*totalNumBlocks);
        output.Mesh()->Component("CELLS")->SetAttribute("totalEntries", (ANA_IS3D?9:5)*totalNumBlocks);
        output.Mesh()->Component("CELLS")->SetAttribute("stride", (ANA_IS3D?9:5));
        output.Mesh()->Component("CELL_TYPES")->SetAttribute("numPoints", totalNumBlocks);
        output.Mesh()->Component("CELL_TYPES")->SetAttribute("bufferCount", totalNumBlocks);
        output.Mesh()->Component("CELL_TYPES")->SetAttribute("stride", 1);
        VtkBuffer points(output.Mesh()->Component("DATASET"));
        VtkBuffer edges(output.Mesh()->Component("CELLS"));
        VtkBuffer cellTypes(output.Mesh()->Component("CELL_TYPES"));
        int count = 0;
        for (int i = 0; i < totalNumTrunks; i++) trunks[i]->RecursiveWritePointsToVtk(points, edges, cellTypes, &count);
        output.Write();
    }

    void RefinementBlock::Render(TikzObject* picture)
    {
        Render(picture, NULL);
    }

    void RefinementBlock::Render(TikzObject* picture, DebugTikzDraw_t debugger)
    {
        picture->PushLineType(TikzLineType::solid, 0.06, TikzColor::black);
        for (int i = 0; i < totalNumTrunks; i++)
        {
            trunks[i]->DrawToObject(picture, debugger);
        }
        picture->PopLineType();
    }

}
