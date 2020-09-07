#include <string>
#include <iostream>
#include "PropTreeLib.h"
#include "Anaptric.h"
#include "RefinementTreeNode.h"
#include "Config.h"
#include <cstdlib>
#include "Utils.hx"
#include "DebugTools.hx"
#include <vector>

namespace Anaptric
{
    RefinementTreeNode::RefinementTreeNode(double* hostBounds, char refineType_in, char refineOrientation_in, int level_in, RefinementTreeNode* host_in, RefinementConstraint::RefinementConstraint constraint_in)
    {
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
        for (int d = 0; d < 2*DIM; d++) isOnBoundary[d] = false;
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
    }

    void RefinementTreeNode::InheritDomainBoundaryInfo(void)
    {
        if (host)
        {
            for (int d = 0; d < 2*DIM; d++)
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
        if (host) {for (int d = 0; d<DIM; d++) directionLevels[d] = host->directionLevels[d] + CharBit(refineType, d);}
        else {for (int d = 0; d<DIM; d++) directionLevels[d] = 0;}
    }

    void RefinementTreeNode::DefineBounds(double* hostBounds, char refineType_in, char refineOrientation_in)
    {
        for (int d = 0; d < DIM; d++)
        {
            char iRefined = (refineType_in>>d)&1;
            char iShift   = (refineOrientation_in>>d)&1;
            blockBounds[2*d]   = hostBounds[2*d]   * (1-(iRefined&iShift))  + (iRefined&iShift) *(0.5*(hostBounds[2*d]+hostBounds[2*d+1]));
            blockBounds[2*d+1] = hostBounds[2*d+1] * (1-(iRefined&~iShift)) + (iRefined&~iShift)*(0.5*(hostBounds[2*d]+hostBounds[2*d+1]));
        }
    }

    void RefinementTreeNode::SetRefineLimiter(RefinementLimit_t* limiter_in)
    {
        refineLimiter = limiter_in;
    }

    void RefinementTreeNode::CreateNewNeighbor(RefinementTreeNode* target, int* deltaijk, char isDomainEdge)
    {
        NodeEdge edgeData;
        edgeData.isDomainEdge = isDomainEdge;
        __dloop(edgeData.edgeVector[d] = deltaijk[d]);
        neighbors.insert({target, edgeData});
        if (isDomainEdge)
        {
            for (int d = 0; d < DIM; d++)
            {
                if ((deltaijk[d]!=0) && CharBit(isDomainEdge, d))
                {
                    int directionOfEdge = (deltaijk[d]<0)?(2*d):(2*d+1);
                    isOnBoundary[directionOfEdge] = true;
                }
            }
        }
    }

    int RefinementTreeNode::GetLevel(void)
    {
        return level;
    }

    void RefinementTreeNode::RemoveNeighbor(RefinementTreeNode* target)
    {
        std::map<RefinementTreeNode*, NodeEdge>::iterator it = neighbors.find(target);
        if(it != neighbors.end())
        {
            neighbors.erase(target);
        }
    }

    void RefinementTreeNode::ResolveNewRefinementWithNeighbors(void)
    {
        std::vector<RefinementTreeNode*> nodes;
        std::vector<bool> refineRequired;
        std::vector<char> refineTypes;
        this->Lock();
        for (std::map<RefinementTreeNode*, NodeEdge>::iterator it = neighbors.begin(); it!=neighbors.end(); it++)
        {
            char newNeighborRefinementType;
            if (!it->first->NodeIsLocked())
            {
                refineRequired.push_back(RefineRequiredFromRelationship(this, it->first, it->second, &newNeighborRefinementType));
                refineTypes.push_back(newNeighborRefinementType);
                nodes.push_back(it->first);
            }
        }
        //Manually enumerate neighbors since contents of the std::map change with refinements
        for (size_t i = 0; i < nodes.size(); i++)
        {
            if (refineRequired[i])
            {
                nodes[i]->Refine(refineTypes[i]);
            }
        }
        this->Unlock();
    }
    
    bool RefinementTreeNode::RefineRequiredFromRelationship(RefinementTreeNode* newChildNode, RefinementTreeNode* toBeRefined, NodeEdge relationship, char* newRefTypeOut)
    {
        *newRefTypeOut = 0;
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
                for (int d = 0; d < DIM; d++)
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

    RefinementTreeNode* RefinementTreeNode::RecursiveGetNodeAt(double coords[DIM])
    {
        if (isTerminal) return this;
        else
        {
            char location = GetOctant(blockBounds, coords);
            int index = GetIndexFromOctantAndRefineType(location, subNodeRefinementType);
            if (index>=numSubNodes) __erkill("Error in RecursiveGetNodeAt: index>=numSubNodes");
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
        if ((refineLimiter!=NULL) && (*refineLimiter!=NULL))
        {
            RefinementLimit_t isLimited = *refineLimiter;
            if (isLimited(this)) return;
        }
        char effective = newRefinementType;
        if (!IS3D) effective = newRefinementType&3;
        subNodeRefinementType = effective;
        isTerminal = false;
        deallocSubTrees = true;
        numSubNodes = NumberOfNewSubNodes(effective);
        subNodes = new RefinementTreeNode* [numSubNodes];
        char newRefinementOrientation = 0;
        int basis = GetCoordBasis(effective);
        for (int i = 0; i < numSubNodes; i++)
        {
            newRefinementOrientation = (char)((i&1)*((basis&0x00ff0000)>>16) + ((i&2)>>1)*((basis&0x0000ff00)>>8) + ((i&4)>>2)*((basis&0x000000ff)));
            subNodes[i] = new RefinementTreeNode(blockBounds, newRefinementType, newRefinementOrientation, level+1, this, constraint);
        }
        GenerateNeighborsOfChildAllNodes();
        UpdateNeighborsOfNeighborsToChildNodes(subNodeRefinementType);
        for (std::map<RefinementTreeNode*, NodeEdge>::iterator it = neighbors.begin(); it!=neighbors.end(); it++)
        {
            it->first->RemoveNeighbor(this);
        }
        for (int i = 0; i < numSubNodes; i++)
        {
            subNodes[i]->ResolveNewRefinementWithNeighbors();
        }
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
                    int deltaijk[DIM];
                    GenerateEdgeRelationshipFromOrientations(childOrientation, siblingOrientation, newRefinementType, deltaijk);
                    newChildNode->CreateNewNeighbor(newSiblingNode, deltaijk, 0);
                }
            }
        }
    }

    void RefinementTreeNode::GenerateEdgeRelationshipFromOrientations(char refFrom, char refTo, char refineType, int* dispVector)
    {
        int shuffledIndices[DIM];
        __dloop(dispVector[d] = 0);
        __dloop(shuffledIndices[d] = (int)CharBit(refTo, d) - (int)CharBit(refFrom, d));
        __dloop(dispVector[d] = shuffledIndices[d]);
    }

    void RefinementTreeNode::UpdateNeighborsOfNeighborsToChildNodes(char newRefinementType)
    {
        for (std::map<RefinementTreeNode*, NodeEdge>::iterator it = neighbors.begin(); it!=neighbors.end(); it++)
        {
            //do stuff with it->first based on it->second
            RefinementTreeNode* neighbor = it->first;
            NodeEdge relationship = it->second;
            int newEdgeVec[DIM];

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
            for (int d = 0; d < DIM; d++)
            {
                int currentBasisVector = 1<<d;
                if (CharBit(~orientationConstraintMask, d))
                {
                    indexingBasis = indexingBasis + currentBasisVector<<(4*currentBasisIndex);
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
                bool relationshipIsAnnihilated;
                for (int d = 0; d < DIM; d++)
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

    void RefinementTreeNode::DetermineNeighborClassificationUpdate(RefinementTreeNode* neighbor, RefinementTreeNode* child, int d, bool tangentUpperOrientation, int* newEdgeVec, bool* relationshipIsAnnihilated)
    {
        int edgeIndex = 2*d + (tangentUpperOrientation?1:0);
        int directionComponentChange = tangentUpperOrientation?1:-1;
        bool allEdgeConditionsSatisfied = true;
        RefinementTreeNode* sameLevelNode;

        for (RefinementTreeNode* currentNode = neighbor; (currentNode->directionLevels[d])>=(child->directionLevels[d]); currentNode = currentNode->host)
        {
            if (!(currentNode->host)) __erkill("Error: neighbor classification reference null host.");
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
        return 1<<(((refinementType&1)?1:0) + ((refinementType&2)?1:0) + ((IS3D*(refinementType&4))?1:0));
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
        for (int d = 0; d < 2*DIM; d++) output = output | isOnBoundary[d];
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
                delete subNodes[i];
            }
            delete [] subNodes;
            neighbors.clear();
        }
    }
}
