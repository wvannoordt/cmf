#include <string>
#include <iostream>
#include "PTL.h"
#include "cmf.h"
#include "RefinementTreeNode.h"
#include "BlockIterator.h"
#include "Config.h"
#include <cstdlib>
#include "Utils.hx"
#include "DebugTools.hx"
#include <vector>
#include "VtkFile.h"
#include "CmfError.h"
#include "CmfScreen.h"
#include "StringUtils.h"
#include "Vec.h"
#include "CmfPrint.h"
#include <map>
#include <algorithm>

namespace cmf
{
    RefinementTreeNode::RefinementTreeNode
    (
        double* hostBounds,
        char refineType_in,
        char refineOrientation_in,
        int level_in,
        RefinementTreeNode* host_in,
        RefinementConstraint::RefinementConstraint constraint_in,
        RefinementBlock* rootBlock_in
    )
    {
        rootBlock = rootBlock_in;
        constraint = constraint_in;
        isLocked = false;
        isTerminal = true;
        refineType = refineType_in;
        refineOrientation = refineOrientation_in;
        DefineBounds(hostBounds, refineType_in, refineOrientation_in);
        numSubNodes = 0;
        deallocSubTrees = false;
        level = level_in;
        host = host_in;
        subNodeRefinementType = 0;
        for (int d = 0; d < 2*CMF_DIM; d++) isOnBoundary[d] = false;
        DefineDirectionLevels();
        InheritDomainBoundaryInfo();
        if (host)
        {
            refineLimiter = host->refineLimiter;
        }
        else
        {
            refineLimiter = NULL;
        }
        nodeTag = 0;
    }

    void RefinementTreeNode::InheritDomainBoundaryInfo(void)
    {
        if (host)
        {
            for (int d = 0; d < 2*CMF_DIM; d++)
            {
                int xyz = (d-(d%2))/2;
                bool currentNodeOnDomainRelativeToHost = !((refineType>>xyz)&1);
                currentNodeOnDomainRelativeToHost = currentNodeOnDomainRelativeToHost || (((refineOrientation>>xyz)&1)==(d&1));
                isOnBoundary[d] = currentNodeOnDomainRelativeToHost&&host->isOnBoundary[d];
            }
        }
    }

    void RefinementTreeNode::DefineDirectionLevels(void)
    {
        if (host)
        {
            for (int d = 0; d < CMF_DIM; d++)
            {
                directionLevels[d] = host->directionLevels[d] + CharBit(refineType, d);
            }
        }
        else
        {
            for (int d = 0; d < CMF_DIM; d++)
            {
                directionLevels[d] = 0;
            }
        }
    }

    void RefinementTreeNode::DefineBounds(double* hostBounds, char refineType_in, char refineOrientation_in)
    {
        for (int d = 0; d < CMF_DIM; d++)
        {
            char iRefined = (refineType_in>>d)&1;
            char iShift   = (refineOrientation_in>>d)&1;
            blockBounds[2*d]   = hostBounds[2*d]   * (1-(iRefined&iShift))  + (iRefined&iShift) *(0.5*(hostBounds[2*d]+hostBounds[2*d+1]));
            blockBounds[2*d+1] = hostBounds[2*d+1] * (1-(iRefined&~iShift)) + (iRefined&~iShift)*(0.5*(hostBounds[2*d]+hostBounds[2*d+1]));
        }
    }
    
    Vec3<double> RefinementTreeNode::GetBlockCenter(void)
    {
        Vec3<double> output(0.0);
        for (int d = 0; d < CMF_DIM; d++) output[d] = 0.5*(blockBounds[2*d] + blockBounds[2*d+1]);
        return output;
    }

    void RefinementTreeNode::SetRefineLimiter(NodeFilter_t* limiter_in)
    {
        refineLimiter = limiter_in;
    }

    void RefinementTreeNode::CreateNewNeighbor(RefinementTreeNode* target, int* deltaijk, char isDomainEdge)
    {
        NodeEdge edgeData;
        edgeData.isDomainEdge = isDomainEdge;
        __dloop(edgeData.edgeVector[d] = deltaijk[d]);
        neighbors.push_back({target, edgeData});
        if (isDomainEdge)
        {
            for (int d = 0; d < CMF_DIM; d++)
            {
                if ((deltaijk[d]!=0) && CharBit(isDomainEdge, d))
                {
                    int directionOfEdge = (deltaijk[d]<0)?(2*d):(2*d+1);
                    isOnBoundary[directionOfEdge] = true;
                }
            }
        }
    }
    
    bool RefinementTreeNode::IsSameDimensionsAs(RefinementTreeNode* node)
    {
        bool output = true;
        __dloop(output = (output && (node->directionLevels[d]==this->directionLevels[d])));
        return output;
    }

    bool RefinementTreeNode::IsTerminal(void)
    {
        return isTerminal;
    }

    int RefinementTreeNode::GetLevel(void)
    {
        return level;
    }

    void RefinementTreeNode::RemoveNeighbor(RefinementTreeNode* target)
    {
        for (int i = 0; i < neighbors.size(); i++)
        {
            if (neighbors[i].first==target)
            {
                neighbors.erase(neighbors.begin()+i);
                i--;
            }
        }
    }

    void RefinementTreeNode::ResolveNewRefinementWithNeighbors(int recursiveLevel)
    {
        std::vector<RefinementTreeNode*> nodes;
        std::vector<bool> refineRequired;
        std::vector<char> refineTypes;
        this->Lock();
        for (auto& it: neighbors)
        {
            char newNeighborRefinementType;
            if (!it.first->NodeIsLocked())
            {
                refineRequired.push_back(RefineRequiredFromRelationship(this, it.first, it.second, &newNeighborRefinementType));
                refineTypes.push_back(newNeighborRefinementType);
                nodes.push_back(it.first);
            }
        }
        //Manually enumerate neighbors since contents of the std::map change with refinements
        for (size_t i = 0; i < nodes.size(); i++)
        {
            if (refineRequired[i])
            {
                nodes[i]->RefineRecursive(refineTypes[i], recursiveLevel+1);
            }
        }
        this->Unlock();
    }
    
    std::vector<std::pair<RefinementTreeNode*, NodeEdge>>& RefinementTreeNode::Neighbors(void)
    {
        return neighbors;
    }

    bool RefinementTreeNode::RefineRequiredFromRelationship(RefinementTreeNode* newChildNode, RefinementTreeNode* toBeRefined, NodeEdge relationship, char* newRefTypeOut)
    {
        *newRefTypeOut = 0;
        bool* periodicRefine = toBeRefined->rootBlock->periodicRefinement;
        bool noRefineBecauseOfPeriodicity = false;
        for (int d = 0; d < CMF_DIM; d++)
        {
            if (CharBit(relationship.isDomainEdge, d)&&!(periodicRefine[d]))
            {
                noRefineBecauseOfPeriodicity = true;
            }
        }
        if (noRefineBecauseOfPeriodicity) return false;
        switch (newChildNode->constraint)
        {
            case RefinementConstraint::free:
            {
                *newRefTypeOut = 0;
                return false;
            }
            case RefinementConstraint::factor2CompletelyConstrained:
            {
                char output = 0;
                bool anyFactorTwo = false;
                for (int d = 0; d < CMF_DIM; d++)
                {
                    if (__d_abs(newChildNode->directionLevels[d] - toBeRefined->directionLevels[d])>1)
                    {
                        anyFactorTwo = true;
                        SetCharBit(output, d, 1);
                    }
                }
                *newRefTypeOut = output;
                return anyFactorTwo;
            }
            case RefinementConstraint::factor2PartiallyConstrained:
            {
                CmfError("RefinementConstraint::factor2PartiallyConstrained not implemented");
                return false;
            }
        }
        return false;
    }

    bool RefinementTreeNode::NodeIsLocked(void)
    {
        return isLocked;
    }

    void RefinementTreeNode::Lock(void)
    {
        isLocked = true;
    }

    void RefinementTreeNode::Unlock(void)
    {
        isLocked = false;
    }

    RefinementTreeNode* RefinementTreeNode::RecursiveGetNodeAt(double coords[CMF_DIM])
    {
        if (isTerminal) return this;
        else
        {
            char location = GetOctant(blockBounds, coords);
            int index = GetIndexFromOctantAndRefineType(location, subNodeRefinementType);
            if (index>=numSubNodes) CmfError("Error in RecursiveGetNodeAt: index>=numSubNodes");
            return subNodes[index]->RecursiveGetNodeAt(coords);
        }
    }

    int RefinementTreeNode::GetIndexFromOctantAndRefineType(char location, char refinementType)
    {
        int basis = GetInvCoordBasis(refinementType);
        char vec = refinementType&location;
        int output = (int)((vec&1)*((basis&0x00ff0000)>>16) + ((vec&2)>>1)*((basis&0x0000ff00)>>8) + ((vec&4)>>2)*((basis&0x000000ff)));
        return output;
    }

    void RefinementTreeNode::RefineRandom()
    {
        if (isTerminal)
        {
            char t = (char)(1+RandomInt(7));
            Refine(t);
        }
        else
        {
            for (int i = 0; i < numSubNodes; i++)
            {
                subNodes[i]->RefineRandom();
            }
        }
    }
    
    void RefinementTreeNode::Refine(char newRefinementType)
    {
        RefineRecursive(newRefinementType, 0);
    }

    void RefinementTreeNode::RefineRecursive(char newRefinementType, int recursiveLevel)
    {
        //do nothing if the node is not a terminal node
        if (!this->IsTerminal()) return;
        //do nothing if the refinement limiter is hit
        if ((refineLimiter!=NULL) && (*refineLimiter!=NULL))
        {
            NodeFilter_t isLimited = *refineLimiter;
            if (isLimited(this)) return;
        }
        char effective = newRefinementType;
        //reduce dimension if needed
        if (!CMF_IS3D) effective = newRefinementType&3;
        subNodeRefinementType = effective;
        //This is no longer a terminal node
        isTerminal = false;
        deallocSubTrees = true;
        numSubNodes = NumberOfNewSubNodes(effective);
        subNodes = new RefinementTreeNode*[numSubNodes];
        char newRefinementOrientation = 0;
        //Get the basis for the child numbering
        int basis = GetCoordBasis(effective);
        for (int i = 0; i < numSubNodes; i++)
        {
            newRefinementOrientation = (char)((i&1)*((basis&0x00ff0000)>>16) + ((i&2)>>1)*((basis&0x0000ff00)>>8) + ((i&4)>>2)*((basis&0x000000ff)));
            subNodes[i] = new RefinementTreeNode(blockBounds, newRefinementType, newRefinementOrientation, level+1, this, constraint, rootBlock);
            //Add the node to the list of all nodes in the base tree
            rootBlock->RegisterNewChildNode(subNodes[i]);
        }
        rootBlock->RegisterNewParentNode(this);
        
        //new child nodes need neighbor relationships
        GenerateNeighborsOfChildAllNodes();
        
        //The current node's neghbors should no longer have a neighbor relationship with this node, but rather with its children
        UpdateNeighborsOfNeighborsToChildNodes(subNodeRefinementType);
        
        //Delete any duplicate neighbor relationships
        for (int i = 0; i < numSubNodes; i++)
        {
            subNodes[i]->DeleteDuplicateNeighbors();
        }
        
        //Remove all neighbors of this node
        for (auto& it: neighbors)
        {
            it.first->RemoveNeighbor(this);
            it.first->DeleteDuplicateNeighbors();
        }
        //Recursively loop through neighboring nodes to check if the refinement constraint is violated
        for (int i = 0; i < numSubNodes; i++)
        {
            subNodes[i]->ResolveNewRefinementWithNeighbors(recursiveLevel);
        }
    }
    
    int RefinementTreeNode::GetOrientationComponent(int component)
    {
        return (int)(CharBit(refineOrientation, component));
    }
    
    void RefinementTreeNode::DeleteDuplicateNeighbors(void)
    {
        //Build a map of indices of each neighbor node in the neighbor array        
        std::map<RefinementTreeNode*, std::vector<int>> neighborCounts;
        int i = 0;
        for (auto& p:neighbors)
        {
            auto node = p.first;
            if (neighborCounts.find(node) == neighborCounts.end()) neighborCounts.insert({node, std::vector<int>()});
            neighborCounts[node].push_back(i);
            i++;
        }
        
        //create a list of indices to remove: a neighbor relationship is removed if the node pointer and the edge vector are the same
        std::vector<int> toRemove;
        for (auto& p:neighborCounts)
        {
            if (p.second.size() > 1)
            {
                std::map<NodeEdge, int> foundEdges;
                for (auto idx:p.second)
                {
                    if (foundEdges.find(neighbors[idx].second) == foundEdges.end()) foundEdges.insert({neighbors[idx].second, 0});
                    else { toRemove.push_back(idx); }
                }
            }
        }
        
        //sort in descending order to ensure that indices are valid through removal
        std::sort(toRemove.begin(), toRemove.end(), std::greater<int>());
        
        //remove at the appropriate indices
        for (auto idx:toRemove)
        {
            neighbors.erase(neighbors.begin() + idx);
        }
    }
    
    Vec3<int> RefinementTreeNode::GetDirectionLevels(void)
    {
        Vec3<int> output(0);
        for (int i = 0; i < CMF_DIM; i++)
        {
            output[i] = directionLevels[i];
        }
        return output;
    }
    
    void RefinementTreeNode::ReadFromFile(ParallelFile& file)
    {
        std::string compatError = "RefinementTreeNode::ReadFromFile compatibility error expecting \"{}\" value of \"{}\", but found \"{}\". Filename: " + file.OpenFileName();
        int readLevel = 0;
        int readRefineType = 0;
        int readNumSubNodes = 0;
        RefinementConstraint::RefinementConstraint nativeConstraint = this->constraint;
        // Reset the refinement constraint to free for now: assume that the incoming mesh has a similar refinment constraint
        this->constraint = RefinementConstraint::free;
        strunformat(file.Read(), "<{}::{}::{}>", readLevel, readRefineType, readNumSubNodes);
        if (readRefineType!=0)
        {
            this->Refine((char)readRefineType);
            //This probably is not the best place to call this: need to think about this in the future.
            this->rootBlock->PostRefinementCallbacks();
            for (int i = 0; i < numSubNodes; i++)
            {
                subNodes[i]->ReadFromFile(file);
            }
        }
        this->constraint = nativeConstraint;
        if (this->level != readLevel) CmfError(strformat(compatError, "level", this->level, readLevel));
    }
    
    void RefinementTreeNode::WriteToFile(ParallelFile& file)
    {
        file.Write(strformat("<{}::{}::{}>", this->GetLevel(), (int)subNodeRefinementType, numSubNodes));
        for (int i = 0; i < numSubNodes; i++)
        {
            subNodes[i]->WriteToFile(file);
        }
    }
    
    void RefinementTreeNode::PrintNeighbors(void)
    {
        print("Node", this, "located at", this->GetBlockCenter(), "has", neighbors.size(), "neighbors, and here they are:");
        for (auto& p:neighbors)
        {
            Vec<int, CMF_DIM> evec(&(p.second.edgeVector[0]));
            print(p.first, "at", p.first->GetBlockCenter(), "in direction", evec);
        }
        print("Good luck.");
    }

    void RefinementTreeNode::GenerateNeighborsOfChildAllNodes(void)
    {        
        //All siblings share a neighboring relationship
        for (int i = 0; i < numSubNodes; i++)
        {
            RefinementTreeNode* newChildNode = subNodes[i];
            for (int j = 0; j < numSubNodes; j++)
            {
                if (i!=j)
                {
                    RefinementTreeNode* newSiblingNode = subNodes[j];
                    char childOrientation = newChildNode->refineOrientation;
                    char siblingOrientation = newSiblingNode->refineOrientation;
                    char newRefinementType = subNodeRefinementType;
                    int deltaijk[CMF_DIM];
                    GenerateEdgeRelationshipFromOrientations(childOrientation, siblingOrientation, newRefinementType, deltaijk);
                    newChildNode->CreateNewNeighbor(newSiblingNode, deltaijk, 0);
                }
            }
        }
    }
    
    int RefinementTreeNode::GetHashableValue(void)
    {
        int output = level + refineOrientation*directionLevels[0] + refineType*directionLevels[1];
#if(CMF_IS3D)
        output += numSubNodes*directionLevels[2];
#endif
        return output;
    }

    void RefinementTreeNode::GenerateEdgeRelationshipFromOrientations(char refFrom, char refTo, char refineType, int* dispVector)
    {
        int shuffledIndices[CMF_DIM];
        __dloop(dispVector[d] = 0);
        __dloop(shuffledIndices[d] = (int)CharBit(refTo, d) - (int)CharBit(refFrom, d));
        __dloop(dispVector[d] = shuffledIndices[d]);
    }

    int RefinementTreeNode::NumberOfNeighbors(void)
    {
        return neighbors.size();
    }

    void RefinementTreeNode::UpdateNeighborsOfNeighborsToChildNodes(char newRefinementType)
    {
        for (auto& it: neighbors)
        {
            //do stuff with it->first based on it->second
            RefinementTreeNode* neighbor = it.first;
            NodeEdge relationship = it.second;
            int newEdgeVec[CMF_DIM];

            //0 -> any bit value allowed
            //1 -> bit value must have the value represented in orientationConstraintValues
            char orientationConstraintMask = 0;
            char orientationConstraintValues = 0;
            __dloop(SetCharBit(orientationConstraintMask, d, relationship.edgeVector[d]!=0));
            __dloop(SetCharBit(orientationConstraintValues, d, relationship.edgeVector[d]>0));
            int numCandidateChildren = 1;
            __dloop(numCandidateChildren = numCandidateChildren << CharBit(~orientationConstraintMask, d));
            int indexingBasis = 0;
            int currentBasisIndex = 0;
            for (int d = 0; d < CMF_DIM; d++)
            {
                int currentBasisVector = 1<<d;
                if (CharBit(~orientationConstraintMask, d))
                {
                    indexingBasis = indexingBasis | (currentBasisVector<<(8*currentBasisIndex));
                    currentBasisIndex++;
                }
            }
            for (int i = 0; i < numCandidateChildren; i++)
            {
                //There will need to be changes here.
                __dloop(newEdgeVec[d] = -relationship.edgeVector[d]);
                char orientationFromBasis = BasisEval(indexingBasis, (char)i);
                char orientation = (orientationFromBasis&~orientationConstraintMask)|(orientationConstraintValues&orientationConstraintMask);
                int orientationToIdxBasis = GetInvCoordBasis(newRefinementType);
                int idx = GetIndexFromOctantAndRefineType(orientation, newRefinementType);
                bool relationshipIsAnnihilated = false;
                for (int d = 0; d < CMF_DIM; d++)
                {
                    
                    bool isUpperOrientationInDirection = subNodes[idx]->SharesEdgeWithHost(2*d+1);
                    bool relationshipStableFromOrientation = (relationship.edgeVector[d]==(isUpperOrientationInDirection?1:-1));
                    bool edgeVectorMightBeReduced = (CharBit(newRefinementType, d)==1);
                    edgeVectorMightBeReduced = edgeVectorMightBeReduced && (subNodes[idx]->directionLevels[d]<=neighbor->directionLevels[d]);
                    edgeVectorMightBeReduced = edgeVectorMightBeReduced && (!relationshipStableFromOrientation);
                    if (edgeVectorMightBeReduced)
                    {
                        relationshipIsAnnihilated = false;                        
                        DetermineNeighborClassificationUpdate(neighbor, subNodes[idx], d, isUpperOrientationInDirection, newEdgeVec, &relationshipIsAnnihilated);
                    }
                }
                if (!relationshipIsAnnihilated)
                {
                    neighbor->CreateNewNeighbor(subNodes[idx], newEdgeVec, relationship.isDomainEdge);
                    __dloop(newEdgeVec[d]*=-1);
                    subNodes[idx]->CreateNewNeighbor(neighbor, newEdgeVec, relationship.isDomainEdge);
                }
            }
        }
    }

    void RefinementTreeNode::RecursiveWritePointsToVtk(VtkBuffer& points, VtkBuffer& edges, VtkBuffer& cellTypes, int* num)
    {
        RecursiveWritePointsToVtk(points, edges, cellTypes, num, [](RefinementTreeNode*){return true;});
    }

    void RefinementTreeNode::RecursiveWritePointsToVtk(VtkBuffer& points, VtkBuffer& edges, VtkBuffer& cellTypes, int* num, NodeFilter_t filter)
    {
        if (isTerminal)
        {
            if (filter(this))
            {
                WriteBlockDataToVtkBuffers(points, edges, cellTypes, num);
            }
        }
        else
        {
            for (int i = 0; i < numSubNodes; i++)
            {
                subNodes[i]->RecursiveWritePointsToVtk(points, edges, cellTypes, num, filter);
            }
        }
    }
    
    void RefinementTreeNode::WriteBlockDataToVtkBuffers(VtkBuffer& points, VtkBuffer& edges, VtkBuffer& cellTypes, int* num)
    {
        edges << (CMF_IS3D?8:4);
        for (char i = 0; i < (1<<CMF_DIM); i++)
        {
            double x = blockBounds[0+CharBit(i, 0)];
            double y = blockBounds[2+CharBit(i, 1)];
            double z = 0.0;
            if (CMF_IS3D) z = blockBounds[4+CharBit(i, 2)];
            points << x;
            points << y;
            points << z;
            edges << (*num);
            (*num) = (*num)+1;
        }
        if (CMF_IS3D) {cellTypes << VtkCellType::voxel;}
        else {cellTypes << VtkCellType::pixel;}
    }
    
    double RefinementTreeNode::MaxEdgeLength(void)
    {
        double dx = blockBounds[1] - blockBounds[0];
        double dy = blockBounds[3] - blockBounds[2];
        double maxout = (dx>dy)?dx:dy;
#if(CMF_IS3D)
        double dz = blockBounds[5] - blockBounds[4];
        maxout = (maxout>dz)?maxout:dz;
#endif
        return maxout;
    }

    void RefinementTreeNode::RecursiveCountTerminal(int* totalNumBlocks)
    {
        RecursiveCountTerminal(totalNumBlocks, [](RefinementTreeNode*){return true;});
    }
    
    void RefinementTreeNode::RecursiveCountTerminal(int* totalNumBlocks, NodeFilter_t filter)
    {
        if (isTerminal)
        {
            if (filter(this))
            {
                (*totalNumBlocks)++;
            }
        }
        else
        {
            for (int i = 0; i < numSubNodes; i++)
            {
                subNodes[i]->RecursiveCountTerminal(totalNumBlocks, filter);
            }
        }
    }

    void RefinementTreeNode::DetermineNeighborClassificationUpdate(RefinementTreeNode* neighbor, RefinementTreeNode* child, int d, bool tangentUpperOrientation, int* newEdgeVec, bool* relationshipIsAnnihilated)
    {
        int edgeIndex = 2*d + (tangentUpperOrientation?1:0);
        int directionComponentChange = tangentUpperOrientation?1:-1;
        bool allEdgeConditionsSatisfied = true;
        RefinementTreeNode* sameLevelNode;

        for (RefinementTreeNode* currentNode = neighbor; (currentNode->directionLevels[d])>=(child->directionLevels[d]); currentNode = currentNode->host)
        {
            if (!(currentNode->host)) CmfError("Error: neighbor classification referenced null host.");
            if (currentNode->directionLevels[d]!=child->directionLevels[d])
            {
                allEdgeConditionsSatisfied = allEdgeConditionsSatisfied && currentNode->SharesEdgeWithHost(edgeIndex);
            }
            sameLevelNode = currentNode;
        }
        bool matchedOnFinalLevel = sameLevelNode->SharesEdgeWithHost(edgeIndex);
        if (matchedOnFinalLevel)
        {
            *relationshipIsAnnihilated = false;
            return;
        }
        else
        {
            if (allEdgeConditionsSatisfied)
            {
                newEdgeVec[d] += directionComponentChange;
            }
            else
            {
                *relationshipIsAnnihilated = !matchedOnFinalLevel;
                return;
            }
        }
    }

    bool RefinementTreeNode::SharesEdgeWithHost(int edgeIndex)
    {
        if (!host) return false;
        int d = (edgeIndex - edgeIndex%2) / 2;
        int upper = edgeIndex%2;
        if (CharBit(~refineType, d)) return true;
        return (CharBit(refineOrientation, d)==upper);
    }

    int RefinementTreeNode::GetCoordBasis(char refinementType)
    {
        int permOrders = 0x01220100;
        char permutationOrder = (permOrders>>(4*refinementType))&0x0000000f;
        int basis = 0x04010204;
        for (char n = 0; n < permutationOrder; n++)
        {
            basis = (basis << 8) + ((basis&0x00ff0000) >> 16);
        }
        return basis;
    }
    int RefinementTreeNode::GetInvCoordBasis(char refinementType)
    {
        int permOrders = 0x01220100;
        char permutationOrder = (permOrders>>(4*refinementType))&0x0000000f;
        permutationOrder = 2*permutationOrder;
        int basis = 0x04010204;
        for (char n = 0; n < permutationOrder; n++)
        {
            basis = (basis << 8) + ((basis&0x00ff0000) >> 16);
        }
        return basis;
    }
    int RefinementTreeNode::NumberOfNewSubNodes(char refinementType)
    {
        return 1<<(((refinementType&1)?1:0) + ((refinementType&2)?1:0) + ((CMF_IS3D*(refinementType&4))?1:0));
    }

    void RefinementTreeNode::DrawToObject(TikzObject* picture, DebugTikzDraw_t debugger)
    {
        double shrink = 0.0;
        double x1 = blockBounds[0]+shrink;
        double y1 = blockBounds[2]+shrink;
        double x2 = blockBounds[1]-shrink;
        double y2 = blockBounds[3]-shrink;
        if (isTerminal)
        {
            picture->DrawBox(x1, y1, x2, y2);
            if (debugger)
            {
                debugger(picture, this);
            }
        }
        else
        {
            for (int i = 0; i < numSubNodes; i++)
            {
                subNodes[i]->DrawToObject(picture, debugger);
            }
        }
    }

    bool RefinementTreeNode::IsAnyDomainBoundary(void)
    {
        bool output = false;
        for (int d = 0; d < 2*CMF_DIM; d++) output = output | isOnBoundary[d];
        return output;
    }

    RefinementTreeNode::~RefinementTreeNode(void)
    {
        this->Destroy();
    }

    double* RefinementTreeNode::GetBlockBounds(void)
    {
        return blockBounds;
    }

    void RefinementTreeNode::Destroy(void)
    {
        if (deallocSubTrees)
        {
            deallocSubTrees = false;
            for (int i = 0; i < numSubNodes; i++)
            {
                subNodes[i]->Destroy();
                WriteLine(8, "Delete block " + PtrToStr(subNodes[i]));
                delete subNodes[i];
            }
            delete [] subNodes;
            neighbors.clear();
        }
    }
}
