#include <string>
#include <iostream>
#include "PropTreeLib.h"
#include "gTree.h"
#include "RefinementTreeNode.h"
#include "Config.h"
#include <cstdlib>
#include "Utils.hx"

namespace gTree
{
    RefinementTreeNode::RefinementTreeNode(double* hostBounds, char refineType_in, char refineOrientation_in)
    {
        isTerminal = true;
        refineType = refineType_in;
        refineOrientation = refineOrientation_in;
        DefineBounds(hostBounds, refineType_in, refineOrientation_in);
        numSubNodes = 0;
        deallocSubTrees = false;
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

    void RefinementTreeNode::Refine(char newRefinementType)
    {
        if (isTerminal)
        {
            RefineLocal(newRefinementType);
        }
        else
        {
            for (int i = 0; i < numSubNodes; i++)
            {
                subNodes[i]->Refine(newRefinementType);
            }
        }
    }
    void RefinementTreeNode::RefineRandom()
    {
        if (isTerminal)
        {
            char t = (char)(1+RandomInt(7));
            RefineLocal(t);
        }
        else
        {
            for (int i = 0; i < numSubNodes; i++)
            {
                char t = (char)(1+RandomInt(7));
                subNodes[i]->Refine(t);
            }
        }
    }

    void RefinementTreeNode::RefineLocal(char newRefinementType)
    {
        char effective = newRefinementType;
        if (!IS3D) effective = newRefinementType&3;
        isTerminal = false;
        deallocSubTrees = true;
        numSubNodes = 1<<(((effective&1)?1:0) + ((effective&2)?1:0) + ((IS3D*(effective&4))?1:0));
        subNodes = new RefinementTreeNode* [numSubNodes];
        char newRefinementOrientation = 0;
        int permOrders = 0x01220100;
        char permutationOrder = (permOrders>>(4*effective))&0x0000000f;
        int basis = 0x04010204;
        for (char n = 0; n < permutationOrder; n++)
        {
            basis = (basis << 8) + ((basis&0x00ff0000) >> 16);
        }
        for (int i = 0; i < numSubNodes; i++)
        {
            newRefinementOrientation = (char)((i&1)*((basis&0x00ff0000)>>16) + ((i&2)>>1)*((basis&0x0000ff00)>>8) + ((i&4)>>2)*((basis&0x000000ff)));
            subNodes[i] = new RefinementTreeNode(blockBounds, newRefinementType, newRefinementOrientation);
        }
    }

    void RefinementTreeNode::DrawToObject(TikzObject* picture)
    {
        if (isTerminal)
        {
            picture->DrawBox(blockBounds[0], blockBounds[2], blockBounds[1], blockBounds[3]);
        }
        else
        {
            for (int i = 0; i < numSubNodes; i++)
            {
                subNodes[i]->DrawToObject(picture);
            }
        }
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
        }
    }
}
