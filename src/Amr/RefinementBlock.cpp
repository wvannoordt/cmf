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
    
    void RefinementBlock::OutputNeighborsToFile(std::string filename)
    {
        std::ofstream myfile;
        myfile.open(filename.c_str());
        myfile << "CMF AMR Neighbor information" << std::endl;
        for (const auto node: allNodes)
        {
            auto& neighs = node->Neighbors();
            for (auto p:neighs)
            {
                auto neighNode = p.first;
                auto neighEdge = p.second;
                cmf::Vec3<int> edgeVec(neighEdge.edgeVector);
                myfile << edgeVec << std::endl;
            }
        }
        myfile.close();
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
            RegisterNewChildNode(trunks[i]);
            trunks[i]->SetRefineLimiter(&refineLimiter);
            for (int d = 0; d < CMF_DIM; d++)
            {
                trunks[i]->GetAmrPosition(2*d).partition = idx[d];
                trunks[i]->GetAmrPosition(2*d).bits = 0;
                trunks[i]->GetAmrPosition(2*d+1).partition = (idx[d]+1);
                trunks[i]->GetAmrPosition(2*d+1).bits = 0;
            }
        }
        //This is added because it is assumed that any post-refinemet callback object is registered afterwards
        newChildNodes.clear();
        newParentNodes.clear();
        
        //Create the neighbor relationships for the trunk nodes
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
    
    void RefinementBlock::RegisterNewChildNode(RefinementTreeNode* newChild)
    {
        allNodes.push_back(newChild);
        this->AugmentHash(newChild->GetHashableValue());
        newChildNodes.push_back(newChild);
    }
    
    void RefinementBlock::RegisterNewParentNode(RefinementTreeNode* newParent)
    {
        this->AugmentHash(newParent->GetHashableValue());
        newParentNodes.push_back(newParent);
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
        WriteLine(3, strformat("Calling post-refinement callbacks on {} objects: {} new child nodes, {} new parent nodes", postRefinementCallbackObjects.size(), newChildNodes.size(), newParentNodes.size()));
        for (int i = 0; i < postRefinementCallbackObjects.size(); i++)
        {
            postRefinementCallbackObjects[i]->OnPostRefinementCallback(newChildNodes, newParentNodes);
        }
        newChildNodes.clear();
        newParentNodes.clear();
    }
    
    void RefinementBlock::RefineNodes(std::vector<RefinementTreeNode*>& nodes, char refineType)
    {
        std::vector<char> vrefineTypes;
        vrefineTypes.resize(nodes.size(), refineType);
        this->RefineNodes(nodes, vrefineTypes);
    }
    
    void RefinementBlock::RefineNodes(std::vector<RefinementTreeNode*>& nodes, std::vector<char>& refineTypes)
    {
        if (nodes.size() != refineTypes.size())
        {
            CmfError(strformat("Attempted call to RefinementBlock::RefineNodes with inconsistent node list size ({}) and refine list size ({})", nodes.size(), refineTypes.size()));
        }
        int i = 0;
        for (auto& n:nodes)
        {
            if (n->IsTerminal())
            {
                n->Refine(refineTypes[i++]);
            }
        }
        this->PostRefinementCallbacks();
    }

    void RefinementBlock::RefineAt(double coords[CMF_DIM], char refinementType)
    {
        RefinementTreeNode* target = GetNodeAt(coords);
        if (target)
        {
            target->Refine(refinementType);
        }
    }
    
    RefinementTreeNode* RefinementBlock::GetNodeAt(Vec3<double>& coords)
    {
        double c[CMF_DIM];
        for (int i = 0; i < CMF_DIM; i++) c[i] = coords.v[i];
        return this->GetNodeAt(c);
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
                delete trunks[i];
            }
            delete[] trunks;
        }
    }
    
    void RefinementBlock::ReadFromFile(ParallelFile& file)
    {
        std::string synchError = "RefinementBlock::ReadFromFile synchronization error, expecting line \"{}\", but found \"{}\". Filename: " + file.OpenFileName();
        std::string compatError = "RefinementBlock::ReadFromFile compatibility error, expecting \"{}\" value of \"{}\", but found \"{}\". Filename: " + file.OpenFileName();
        std::string line = "";
        if ((line=file.Read()) != "<tree>") CmfError(strformat(synchError, "<tree>", line));
        
        size_t readHash = 0;
        int readDim = 0;
        int readBlockDims[3] = {0};
        strunformat((line=file.Read()), "Hash: {}", readHash);
        strunformat((line=file.Read()), "Dim: {}", readDim);
        strunformat((line=file.Read()), "Blocks: ({}, {}, {})", readBlockDims[0], readBlockDims[1], readBlockDims[2]);
        if (readDim != CMF_DIM) CmfError(strformat(compatError, "Dim", CMF_DIM, readDim));
        if (readBlockDims[0] != blockDim[0]) CmfError(strformat(compatError, "Blocks[0]", blockDim[0], readBlockDims[0]));
        if (readBlockDims[1] != blockDim[1]) CmfError(strformat(compatError, "Blocks[1]", blockDim[1], readBlockDims[1]));
#if (CMF_IS3D)
        if (readBlockDims[2] != blockDim[2]) CmfError(strformat(compatError, "Blocks[2]", blockDim[2], readBlockDims[2]));
#endif
        if ((line=file.Read()) != "<treedata>") CmfError(strformat(synchError, "<treedata>", line));
        for (int i = 0; i < totalNumTrunks; i++)
        {
            trunks[i]->ReadFromFile(file);
        }
        if ((line=file.Read()) != "</treedata>") CmfError(strformat(synchError, "</treedata>", line));
        if ((line=file.Read()) != "</tree>") CmfError(strformat(synchError, "</tree>", line));
        if (readHash != this->GetHash()) CmfError(strformat("Critical read error from file {}: mesh hash function mismatch, refinement pattern of mesh after read does not match the one in the file", file.OpenFileName()));
    }
    
    void RefinementBlock::WriteToFile(ParallelFile& file)
    {
        file.Write("<tree>");
        file.Write(strformat("Hash: {}", this->GetHash()));
        file.Write(strformat("Dim: {}", CMF_DIM));
        file.Write(strformat("Blocks: ({}, {}, {})", blockDim[0], blockDim[1], CMF_IS3D?blockDim[CMF_DIM-1]:1));
        file.Write("<treedata>");
        for (int i = 0; i < totalNumTrunks; i++)
        {
            trunks[i]->WriteToFile(file);
        }
        file.Write("</treedata>");
        file.Write("</tree>");
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
