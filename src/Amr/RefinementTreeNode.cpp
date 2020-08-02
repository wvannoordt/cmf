#include <string>
#include <iostream>
#include "PropTreeLib.h"
#include "gTree.h"
#include "RefinementTreeNode.h"
#include "Config.h"
#include <cstdlib>
#include "Utils.hx"
#include "DebugTools.hx"

namespace gTree
{
    RefinementTreeNode::RefinementTreeNode(double* hostBounds, char refineType_in, char refineOrientation_in, int level_in, RefinementTreeNode* host_in)
    {
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
    
    void RefinementTreeNode::CreateNewNeighbor(RefinementTreeNode* target, char edgeDirection, bool isDomainEdge)
    {
        NodeEdge edgeData;
        edgeData.edgeDirection = edgeDirection;
        edgeData.isDomainEdge = isDomainEdge;
        neighbors.insert({target, edgeData});
        if (isDomainEdge) isOnBoundary[edgeDirection] = true;
    }
    
    void RefinementTreeNode::ResolveNewRefinementWithNeighbor(RefinementTreeNode* issuer)
    {
        
    }
    
    RefinementTreeNode* RefinementTreeNode::RecursiveGetNodeAt(double coords[DIM])
    {
        if (isTerminal) return this;
        else
        {
            char location = GetOctant(blockBounds, coords);
            int index = GetIndexFromOctantAndRefineType(location, subNodeRefinementType);
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
        char effective = newRefinementType;
        subNodeRefinementType = effective;
        if (!IS3D) effective = newRefinementType&3;
        isTerminal = false;
        deallocSubTrees = true;
        numSubNodes = NumberOfNewSubNodes(effective);
        subNodes = new RefinementTreeNode* [numSubNodes];
        char newRefinementOrientation = 0;
        int basis = GetCoordBasis(effective);
        for (int i = 0; i < numSubNodes; i++)
        {
            newRefinementOrientation = (char)((i&1)*((basis&0x00ff0000)>>16) + ((i&2)>>1)*((basis&0x0000ff00)>>8) + ((i&4)>>2)*((basis&0x000000ff)));
            subNodes[i] = new RefinementTreeNode(blockBounds, newRefinementType, newRefinementOrientation, level+1, this);
        }
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

    void RefinementTreeNode::DrawToObject(TikzObject* picture)
    {
        double shrink = 0.0;
        double x1 = blockBounds[0]+shrink;
        double y1 = blockBounds[2]+shrink;
        double x2 = blockBounds[1]-shrink;
        double y2 = blockBounds[3]-shrink;
        if (isTerminal)
        {
            if (IsAnyDomainBoundary()) picture->FillBox(x1, y1, x2, y2);
            picture->DrawBox(x1, y1, x2, y2);
        }
        else
        {
            for (int i = 0; i < numSubNodes; i++)
            {
                subNodes[i]->DrawToObject(picture);
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
