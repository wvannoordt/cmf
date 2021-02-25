#include <string>
#include <iostream>
#include "PTL.h"
#include "cmf.h"
#include "RefinementBlock.h"
#include "RefinementTreeNode.h"
#include "Config.h"
#include "TikzObject.h"
#include "Utils.hx"
#include "RefinementConstraint.h"
#include "VtkFile.h"
#include "CmfScreen.h"
#include "StringUtils.h"
#include "IPostRefinementCallback.h"
namespace cmf
{
    RefinementBlock::RefinementBlock(int* blockDim_in, double* blockBounds_in, RefinementConstraint::RefinementConstraint constraint_in, bool* periodicRefinement_in)
    {
        periodicRefinement = periodicRefinement_in;
        blockDim = blockDim_in;
        blockBounds = blockBounds_in;
        refinementConstraintType = constraint_in;
        refineLimiter = NULL;
        crashOnQueryOutsideDomain = false;
        srand((unsigned int)time(NULL));
        totalNumTrunks = 1;
        for (int i = 0; i < CMF_DIM; i++) totalNumTrunks*=blockDim[i];
        for (int d = 0; d < CMF_DIM; d++) dx[d] = (blockBounds[2*d+1]-blockBounds[2*d])/blockDim[d];
        __dloop(this->AugmentHash(blockDim[d]));
        this->AugmentHash(totalNumTrunks);
        DefineTrunks();
    }
    
    size_t RefinementBlock::Size(void)
    {
        return allNodes.size();
    }
    
    void RefinementBlock::NoCrashOnQueryOutsideDomain(void)
    {
        crashOnQueryOutsideDomain = true;
    }
    
    size_t RefinementBlock::Size(NodeFilter_t filter)
    {
        size_t count = 0;
        for (size_t i = 0; i < allNodes.size(); i++)
        {
            if (filter(allNodes[i])) count++;
        }
        return count;
    }
    
    std::vector<RefinementTreeNode*>* RefinementBlock::GetAllNodes(void)
    {
        return &allNodes;
    }
    
    RefinementConstraint::RefinementConstraint RefinementBlock::GetRefinementConstraintType(void)
    {
        return refinementConstraintType;
    }
    
    RefinementBlock* RefinementBlock::GetRefinementBlockObject(void)
    {
        return this;
    }
    
    double* RefinementBlock::GetBlockBounds(void)
    {
        return blockBounds;
    }

    RefinementBlock::~RefinementBlock(void)
    {
        this->Destroy();
    }

    void RefinementBlock::DefineTrunks(void)
    {
        deallocTrunks = true;
        trunks = new RefinementTreeNode* [totalNumTrunks];
        double localBounds[2*CMF_DIM];
        int idx[CMF_DIM];        
        for (int i = 0; i < totalNumTrunks; i++)
        {
            Dim2Idx(i, blockDim, idx);
            for (int d = 0; d < CMF_DIM; d++)
            {
                localBounds[2*d]   = blockBounds[2*d]+idx[d]*dx[d];
                localBounds[2*d+1] = blockBounds[2*d]+(idx[d]+1)*dx[d];
            }
            trunks[i] = new RefinementTreeNode(localBounds, 0, 0, 0, NULL, refinementConstraintType, this);
            RegisterNewNode(trunks[i]);
            trunks[i]->SetRefineLimiter(&refineLimiter);
        }
        //This is added because it is assumed that any post-refinemet callback object is registered afterwards
        newlyRefinedNodes.clear();
        for (int i = 0; i < totalNumTrunks; i++)
        {
            int blockIndex[CMF_DIM];
            Dim2Idx(i, blockDim, blockIndex);
            int totalNumNeighbors = CMF_IS3D?27:9;
            int boxdim[CMF_DIM];
            __dloop(boxdim[d] = 3);
            int neighborBlockIndex[CMF_DIM];
            __dloop(neighborBlockIndex[d] = 0);
            for (int k = 0; k < totalNumNeighbors; k++)
            {
                int deltaijk[CMF_DIM];
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
    
    void RefinementBlock::RegisterNewNode(RefinementTreeNode* newNode)
    {
        allNodes.push_back(newNode);
        this->AugmentHash(newNode->GetHashableValue());
        newlyRefinedNodes.push_back(newNode);
    }

    void RefinementBlock::SetRefineLimitCriterion(NodeFilter_t limiter_in)
    {
        refineLimiter = limiter_in;
    }
    
    int RefinementBlock::AddPostRefinementCallbackObject(IPostRefinementCallback* obj)
    {
        postRefinementCallbackObjects.push_back(obj);
        return (postRefinementCallbackObjects.size()-1);
    }
    
    void RefinementBlock::PostRefinementCallbacks(void)
    {
        for (int i = 0; i < postRefinementCallbackObjects.size(); i++)
        {
            postRefinementCallbackObjects[i]->OnPostRefinementCallback(newlyRefinedNodes);
        }
        newlyRefinedNodes.clear();
    }

    void RefinementBlock::RefineAt(double coords[CMF_DIM], char refinementType)
    {
        RefinementTreeNode* target = GetNodeAt(coords);
        if (target)
        {
            target->Refine(refinementType);
        }
    }

    RefinementTreeNode* RefinementBlock::GetNodeAt(double coords[CMF_DIM])
    {
        int idx[CMF_DIM];
        if (PointIsInDomain(coords, idx))
        {
            return trunks[Idx2Dim(blockDim, idx)]->RecursiveGetNodeAt(coords);
        }
        else
        {
            HandleRefinementQueryOutsideDomain(coords);
            return NULL;
        }
    }

    bool RefinementBlock::PointIsInDomain(double coords[CMF_DIM], int* idx)
    {
        bool queryOutsideDomain = false;
        for (int d = 0; d < CMF_DIM; d++)
        {
            idx[d] = (coords[d] - blockBounds[2*d])/(dx[d]);
            queryOutsideDomain = queryOutsideDomain  || (coords[d]<blockBounds[2*d]) || (coords[d]>=blockBounds[2*d+1]);
        }
        return !queryOutsideDomain;
    }

    bool RefinementBlock::PointIsInDomain(double coords[CMF_DIM])
    {
        int idx[CMF_DIM];
        return PointIsInDomain(coords, idx);
    }

    void RefinementBlock::HandleRefinementQueryOutsideDomain(double coords[CMF_DIM])
    {
        if (!crashOnQueryOutsideDomain)
        {            
            std::string queryCoords = "(" + std::to_string(coords[0]);
            for (int i = 1; i < CMF_DIM; i++)
            {
                queryCoords = queryCoords + ", " + std::to_string(coords[i]);
            }
            queryCoords = queryCoords + ")";
            CmfError("Attempted to fetch a node outside the domain with coordiantes " + queryCoords + ".");
        }
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
                WriteLine(8, "Delete block " + PtrToStr(trunks[i]));
                delete trunks[i];
            }
            delete[] trunks;
        }
    }

    //This is for debugging only. For any real VTK output, an externl iterator should be used.
    void RefinementBlock::OutputDebugVtk(std::string filename)
    {
        OutputDebugVtk(filename, [](RefinementTreeNode*){return true;});
    }
    
    //This is for debugging only. For any real VTK output, an externl iterator should be used.
    void RefinementBlock::OutputDebugVtk(std::string filename, NodeFilter_t filter)
    {
        int debug = 0;
        VtkFile output(filename, VtkFormatType::ascii, VtkTopologyType::unstructuredGrid);
        int totalNumBlocks = 0;
        for (int i = 0; i < totalNumTrunks; i++) trunks[i]->RecursiveCountTerminal(&totalNumBlocks, filter);
        output.Mesh()->Component("DATASET")->SetAttribute("numPoints",   (CMF_IS3D?8:4)*totalNumBlocks);
        output.Mesh()->Component("DATASET")->SetAttribute("bufferCount", 3*(CMF_IS3D?8:4)*totalNumBlocks);
        output.Mesh()->Component("DATASET")->SetAttribute("stride", 3);
        output.Mesh()->Component("CELLS")->SetAttribute("numPoints", totalNumBlocks);
        output.Mesh()->Component("CELLS")->SetAttribute("bufferCount", (CMF_IS3D?9:5)*totalNumBlocks);
        output.Mesh()->Component("CELLS")->SetAttribute("totalEntries", (CMF_IS3D?9:5)*totalNumBlocks);
        output.Mesh()->Component("CELLS")->SetAttribute("stride", (CMF_IS3D?9:5));
        output.Mesh()->Component("CELL_TYPES")->SetAttribute("numPoints", totalNumBlocks);
        output.Mesh()->Component("CELL_TYPES")->SetAttribute("bufferCount", totalNumBlocks);
        output.Mesh()->Component("CELL_TYPES")->SetAttribute("stride", 1);
        VtkBuffer points(output.Mesh()->Component("DATASET"));
        VtkBuffer edges(output.Mesh()->Component("CELLS"));
        VtkBuffer cellTypes(output.Mesh()->Component("CELL_TYPES"));
        int count = 0;
        for (int i = 0; i < totalNumTrunks; i++) trunks[i]->RecursiveWritePointsToVtk(points, edges, cellTypes, &count, filter);
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
    
    void RefinementBlock::OutputNodesToVtk(const std::vector<RefinementTreeNode*>& nodeList, std::string filename)
    {
        VtkFile output(filename, VtkFormatType::ascii, VtkTopologyType::unstructuredGrid);
        int totalNumBlocks = nodeList.size();
        output.Mesh()->Component("DATASET")->SetAttribute("numPoints",   (CMF_IS3D?8:4)*totalNumBlocks);
        output.Mesh()->Component("DATASET")->SetAttribute("bufferCount", 3*(CMF_IS3D?8:4)*totalNumBlocks);
        output.Mesh()->Component("DATASET")->SetAttribute("stride", 3);
        output.Mesh()->Component("CELLS")->SetAttribute("numPoints", totalNumBlocks);
        output.Mesh()->Component("CELLS")->SetAttribute("bufferCount", (CMF_IS3D?9:5)*totalNumBlocks);
        output.Mesh()->Component("CELLS")->SetAttribute("totalEntries", (CMF_IS3D?9:5)*totalNumBlocks);
        output.Mesh()->Component("CELLS")->SetAttribute("stride", (CMF_IS3D?9:5));
        output.Mesh()->Component("CELL_TYPES")->SetAttribute("numPoints", totalNumBlocks);
        output.Mesh()->Component("CELL_TYPES")->SetAttribute("bufferCount", totalNumBlocks);
        output.Mesh()->Component("CELL_TYPES")->SetAttribute("stride", 1);
        VtkBuffer points(output.Mesh()->Component("DATASET"));
        VtkBuffer edges(output.Mesh()->Component("CELLS"));
        VtkBuffer cellTypes(output.Mesh()->Component("CELL_TYPES"));
        int dummy = 0;
        for (int i = 0; i < totalNumBlocks; i++)
        {
            nodeList[i]->WriteBlockDataToVtkBuffers(points, edges, cellTypes, &dummy);
        }
        output.Write();
    }
}
