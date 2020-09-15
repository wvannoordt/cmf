#ifndef RefinementTreeNode_H
#define RefinementTreeNode_H

#include <string>
#include "Config.h"
#include "TikzObject.h"
#include <vector>
#include <utility>
#include "RefinementConstraint.h"
#include "VtkFile.h"

namespace Anaptric
{
    struct NodeEdge
    {
        char isDomainEdge;
        int edgeVector[ANA_DIM];
    };

    class RefinementTreeNode;
    typedef bool(*RefinementLimit_t)(RefinementTreeNode*);
    typedef void(*DebugTikzDraw_t)(TikzObject*, RefinementTreeNode*);

    class RefinementTreeNode
    {
        public:
            RefinementTreeNode(double* hostBounds, char refineType_in, char refineOrientation_in, int level_in, RefinementTreeNode* host_in, RefinementConstraint::RefinementConstraint constraint_in);
            ~RefinementTreeNode(void);
            void Destroy(void);
            void RefineRandom();
            void DrawToObject(TikzObject* picture, DebugTikzDraw_t debugger);
            void ResolveNewRefinementWithNeighbors(void);
            void CreateNewNeighbor(RefinementTreeNode* target, int* deltaijk, char isDomainEdge);
            void RemoveNeighbor(RefinementTreeNode* target);
            bool IsAnyDomainBoundary(void);
            RefinementTreeNode* RecursiveGetNodeAt(double coords[ANA_DIM]);
            void Refine(char newRefinementType);
            void SetRefineLimiter(RefinementLimit_t* limiter_in);
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
            double blockBounds[2*ANA_DIM];
            RefinementConstraint::RefinementConstraint constraint;
            RefinementTreeNode** subNodes;
            RefinementTreeNode* host;
            int numSubNodes, level;
            int directionLevels[ANA_DIM];
            std::map<RefinementTreeNode*, NodeEdge> neighbors;
            bool isOnBoundary[2*ANA_DIM];
            RefinementLimit_t* refineLimiter;

            friend class NeighborIterator;
    };
}

#endif
