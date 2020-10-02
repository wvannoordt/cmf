#ifndef RefinementTreeNode_H
#define RefinementTreeNode_H

#include <string>
#include "AmrFcnTypes.h"
#include "Config.h"
#include "TikzObject.h"
#include <vector>
#include <utility>
#include "RefinementBlock.h"
#include "RefinementConstraint.h"
#include "BlockIterator.h"
#include "VtkFile.h"

namespace cmf
{

    struct NodeEdge
    {
        char isDomainEdge;
        int edgeVector[CMF_DIM];
    };

    class RefinementBlock;
    class RefinementTreeNode
    {
        public:
            RefinementTreeNode
            (
                double* hostBounds,
                char refineType_in,
                char refineOrientation_in,
                int level_in,
                RefinementTreeNode* host_in,
                RefinementConstraint::RefinementConstraint constraint_in,
                RefinementBlock* rootBlock_in
            );
            ~RefinementTreeNode(void);
            void Destroy(void);
            void RefineRandom();
            void DrawToObject(TikzObject* picture, DebugTikzDraw_t debugger);
            void ResolveNewRefinementWithNeighbors(void);
            void CreateNewNeighbor(RefinementTreeNode* target, int* deltaijk, char isDomainEdge);
            void RemoveNeighbor(RefinementTreeNode* target);
            bool IsAnyDomainBoundary(void);
            RefinementTreeNode* RecursiveGetNodeAt(double coords[CMF_DIM]);
            void Refine(char newRefinementType);
            void SetRefineLimiter(NodeFilter_t* limiter_in);
            int GetLevel(void);
            bool SharesEdgeWithHost(int edgeIndex);
            double* GetBlockBounds(void);
            void RecursiveCountTerminal(int* totalNumBlocks);
            void RecursiveWritePointsToVtk(VtkBuffer& points, VtkBuffer& edges, VtkBuffer& cellTypes, int* num);
        private:
            void Lock(void);
            void Unlock(void);
            bool NodeIsLocked(void);
            static bool RefineRequiredFromRelationship(RefinementTreeNode* newChildNode, RefinementTreeNode* toBeRefined, NodeEdge relationship, char* newRefTypeOut);
            void GenerateEdgeRelationshipFromOrientations(char refFrom, char refTo, char refineType, int* dispVector);
            void GenerateNeighborsOfChildAllNodes(void);
            void UpdateNeighborsOfNeighborsToChildNodes(char newRefinementType);
            void InheritDomainBoundaryInfo(void);
            void DefineDirectionLevels(void);
            int GetIndexFromOctantAndRefineType(char location, char refinementType);
            int NumberOfNewSubNodes(char refinementType);
            int GetCoordBasis(char refinementType);
            int GetInvCoordBasis(char refinementType);
            void DefineBounds(double* hostBounds, char refineType_in, char refineOrientation_in);
            void DetermineNeighborClassificationUpdate(RefinementTreeNode* neighbor, RefinementTreeNode* child, int d, bool tangentUpperOrientation, int* newEdgeVec, bool* relationshipIsAnnihilated);
            char refineType, refineOrientation;
            bool isTerminal, deallocSubTrees, isLocked;
            char subNodeRefinementType;
            double blockBounds[2*CMF_DIM];
            RefinementConstraint::RefinementConstraint constraint;
            RefinementTreeNode** subNodes;
            RefinementTreeNode* host;
            int numSubNodes, level;
            int directionLevels[CMF_DIM];
            std::map<RefinementTreeNode*, NodeEdge> neighbors;
            bool isOnBoundary[2*CMF_DIM];
            NodeFilter_t* refineLimiter;
            RefinementBlock* rootBlock;
            int iteratorIndex;
            friend class NeighborIterator;
            friend class BlockIterator;
    };
}

#endif
