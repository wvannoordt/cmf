#ifndef RefinementTreeNode_H
#define RefinementTreeNode_H

#include <string>
#include "Config.h"
#include "TikzObject.h"
#include <vector>
#include <utility>

namespace gTree
{
    struct NodeEdge
    {
        char isDomainEdge;
        int edgeVector[DIM];
    };

    class RefinementTreeNode;
    typedef bool(*RefinementLimit_t)(RefinementTreeNode*);

    class RefinementTreeNode
    {
        public:
            RefinementTreeNode(double* hostBounds, char refineType_in, char refineOrientation_in, int level_in, RefinementTreeNode* host_in);
            ~RefinementTreeNode(void);
            void Destroy(void);
            void RefineRandom();
            void DrawToObject(TikzObject* picture);
            void ResolveNewRefinementWithNeighbor(RefinementTreeNode* issuer);
            void CreateNewNeighbor(RefinementTreeNode* target, int* deltaijk, char isDomainEdge);
            void RemoveNeighbor(RefinementTreeNode* target);
            bool IsAnyDomainBoundary(void);
            RefinementTreeNode* RecursiveGetNodeAt(double coords[DIM]);
            void Refine(char newRefinementType);
            void SetRefineLimiter(RefinementLimit_t* limiter_in);
            int GetLevel(void);
            bool SharesEdgeWithHost(int edgeIndex);
        private:
            void GenerateEdgeRelationshipFromOrientations(char refFrom, char refTo, char refineType, int* dispVector);
            void GenerateNeighborsOfChildAllNodes(void);
            void UpdateNeighborsOfNeighborsToChildNodes(char newRefinementType);
            void InheritDomainBoundaryInfo(void);
            void DefineDirectionLevels(void);
            void DebugDraw(TikzObject* picture);
            int GetIndexFromOctantAndRefineType(char location, char refinementType);
            int NumberOfNewSubNodes(char refinementType);
            int GetCoordBasis(char refinementType);
            int GetInvCoordBasis(char refinementType);
            void DefineBounds(double* hostBounds, char refineType_in, char refineOrientation_in);
            void DetermineNeighborClassificationUpdate(RefinementTreeNode* neighbor, RefinementTreeNode* child, int d, bool tangentUpperOrientation, int* newEdgeVec, bool* relationshipIsAnnihilated);
            char refineType, refineOrientation;
            bool isTerminal, deallocSubTrees;
            char subNodeRefinementType;
            double blockBounds[2*DIM];
            RefinementTreeNode** subNodes;
            RefinementTreeNode* host;
            int numSubNodes, level;
            int directionLevels[DIM];
            std::map<RefinementTreeNode*, NodeEdge> neighbors;
            bool isOnBoundary[2*DIM];
            RefinementLimit_t* refineLimiter;

    };
}

#endif
