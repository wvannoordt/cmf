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
#include "ParallelFile.h"
#include <algorithm>
#include "Vec.h"
#include "CmfPrint.h"
#include "ExactAmrPosition.h"

namespace cmf
{
    
    typedef uint64_t amrBitType;
    /// @brief Struct that represents the information corresponding to the relationship between two neighboring nodes.
	/// @author WVN
    struct NodeEdge
    {
        /// @brief A char interpreted as a vector of bools that denote whether or not the edge relationship wraps around to the other side of the block
        char isDomainEdge;
        
        /// @brief A normal vector in index space describing the topological relationship between two nodes. E.g. if two nodes share a face in x, then one will have edgeVector = (1, 0, [0]) and the
        /// other will have edgeVector = (-1, 0, [0])
        int edgeVector[CMF_DIM];
        
        /// @brief Determines equality between two NodeEdge Objects
        /// @param n1 the first one
        /// @param n2 the second one
        friend bool operator== ( const NodeEdge &n1, const NodeEdge &n2)
        {
            bool output = true;
            for (int i = 0; i < CMF_DIM; i++) output = (output && (n1.edgeVector[i] == n2.edgeVector[i]));
            return (output && (n1.isDomainEdge == n2.isDomainEdge));
        }
        
        /// @brief Returns a dummy value for the sake of comparison
        /// @param rhs the object to compare to
        inline int BadHash(void) const
        {
            return (1+edgeVector[0]) + 3*(1+edgeVector[1]) + 9*CMF_IS3D*(1+edgeVector[CMF_DIM-1])+27*isDomainEdge;
        }
        
        /// @brief Comparison operator for map insertion
        /// @param rhs the object to compare to
        bool operator <(const NodeEdge& rhs) const
        {
            return BadHash() < rhs.BadHash();
        }
    };
    
    ///@brief Allows streaming of NodeEdge
    static std::ostream & operator<<(std::ostream & os, const NodeEdge & edge)
    {
        os << "{{";
        for (int i = 0; i < CMF_DIM; i++)
        {
            os << edge.edgeVector[i];
            if (i!=CMF_DIM-1) os << ", ";
        }
        os << "}}";
        return os;
    }

    class RefinementBlock;
    /// @brief Class that represents a "node" in the AMR refinement tree. Could also be called a "block". Typically is assigned some chunk of simulation data to be associated with it.
	/// @author WVN
    class RefinementTreeNode
    {
        public:
            /// @brief Constructor for RefinementTreeNode.
            /// \pre This should never be called directly as nodes must belong to a tree object (RefinementBlock).            
            /// @param hostBounds An array of size CMF_DIM describing the bounds of the host RefinementTreeNode, where bounds = (xmin, xmax, ymin, ymax, [zmin, zmax])
            /// @param refineType_in A char interpreted as a 3-vector of bits describing how the host block was refined to produce this one. E.g. refineType_in = 5
            /// means that the host block was refined in x and z, since 5 = 101
            /// @param refineOrientation_in A char interpreted as a 3-vector of bits describing the cartesian orientation with respect to the host RefinementTreeNode.
            /// E.g. if refineType_in = 5 = 101 and refineOrientation_in = 3 = 011 then this node rests on the negative-z face of the host block, and the positive x face. Note
            /// that the second bit is ignored since there is no refinement of the host node in that direction.
            /// @param level_in The refinement level, i.e. the number of refinements that took place to get to this node
            /// @param host_in A pointer to the host node
            /// @param constraint_in A RefinementConstraint enum indicating how refinements are constrained 
            /// @param rootBlock_in A pointer to the containing RefinementBlock that this node is a part of
            /// \see RefinementBlock RefinementBlock Config.h RefinementConstraint.h ResolveNewRefinementWithNeighbors RefineRequiredFromRelationship
        	/// @author WVN
            RefinementTreeNode
            (
                std::vector<double> hostBounds,
                char refineType_in,
                char refineOrientation_in,
                int level_in,
                RefinementTreeNode* host_in,
                RefinementConstraint::RefinementConstraint constraint_in,
                RefinementBlock* rootBlock_in
            );
            
            /// @brief Destructor for RefinementTreeNode. Explicitly calls Destroy. \see Destroy
            /// @author WVN
            ~RefinementTreeNode(void);
            
            /// @brief Explicitly destroys all child nodes recursively and releases resources. Generally should
            /// not be called except in the case of advanced usage.
            /// @author WVN
            void Destroy(void);
            
            /// @brief Recursively randomly refines current block (if terminal) or child blocks (if not terminal) \see isTerminal
            /// @author WVN
            void RefineRandom();
            
            /// @brief Draw the current node to the provided TikZObject if the DebugTikzDraw_t condition is satisfied
            /// @param picture An active TikzObject to be drawn to
            /// @param debugger A DebugTikzDraw_t function pointer that, if returns false on current node, skips the drawing step
            /// \see AmrFcnTypes.h
            /// @author WVN
            void DrawToObject(TikzObject* picture, DebugTikzDraw_t debugger);
            
            /// @brief Checks if, based on the RefinementConstraint of the current node, the neighbors of current node
            /// should be refined. Note that this is called recursively through Refine until the RefinementConstraint is satisfied, so
            /// the call stack can become quite large.
            /// \see RefineRequiredFromRelationship Refine
            /// @param recursiveLevel The level of recursion
            /// @author WVN
            void ResolveNewRefinementWithNeighbors(int recursiveLevel);
            
            /// @brief Creates a neighbor relationship between the current block and target, and creates
            /// a NodeEdge with correspnding data. Does not check to see if the relationship already
            /// exists.
            /// @param target A pointer to the RefinementTreeNode to add as a neighbor
            /// @param deltaijk populates the edgeVector of NodeEdge and chould have size CMF_DIM
            /// @param isDomainEdge Assigned to isDomainEdge of the resulting NodeEdge
            /// \see NodeEdge
            /// @author WVN
            void CreateNewNeighbor(RefinementTreeNode* target, int* deltaijk, char isDomainEdge);
            
            /// @brief Removes a neighbor relationship between the current block and target.
            /// @param target A pointer to the RefinementTreeNode to remove as a neighbor
            /// @author WVN
            void RemoveNeighbor(RefinementTreeNode* target);
            
            /// @brief Returns true if the current node lies on any boundary of the underlying RefinementBlock.
            /// @author WVN
            bool IsAnyDomainBoundary(void);
            
            /// @brief Returns true if the current node has no children, that is, if it is terminal.
            /// @author WVN
            bool IsTerminal(void);
            
            /// @brief Deletes any duplicate neighbor relationships
            /// @author WVN
            void DeleteDuplicateNeighbors(void);
            
            
            /// @brief Returns 1 if this block is in the positive direction along the axis specified by component with respect to the parent,
            /// and 0 otherwise
            /// @author WVN
            int GetOrientationComponent(int component);
            
            /// @brief Returns the refinement orientation
            /// @author WVN
            int GetOrientation(void)
            {
                return refineOrientation;
            }
            
            /// @brief Returns the exact AMR position at the spcified index (xmin, xmax, ymin, ymax, [zmin, zmax])
            /// coords are contained within the bounds
            /// @param idx the index of the desired element
            /// @author WVN
            ExactAmrPosition<amrBitType>& GetAmrPosition(int idx)
            {
                return exactBounds[idx];
            }
            
            /// @brief Returns a pointer to the current node if it is terminal and
            /// coords are contained within the bounds
            /// @param coords An array of size CMF_DIM containing Cartesian coordinates
            /// @author WVN
            RefinementTreeNode* RecursiveGetNodeAt(double coords[CMF_DIM]);
            
            /// @brief Refines the current node according to newRefinementType. Values
            /// of newRefinementType correspond as (1,2,3,4,5,6,7) -> (x, y, xy, z, xz, yz, xyz)
            /// \pre Note that this function might recursively call itself depending on the value of the RefinementConstraint.
            /// Note that GenerateNeighborsOfChildAllNodes, UpdateNeighborsOfNeighborsToChildNodes, and ResolveNewRefinementWithNeighbors
            /// are also called.
            /// @param newRefinementType A char interpreted as a 3-vector of bits indicating refinement directions
            /// @author WVN
            void Refine(char newRefinementType);
            
            /// @brief Sets a hard limit on whether or not the current node can be refined.
            /// If Refine is called on this node, nothing will happen if the function pointer
            /// limiter_in does returns true.
            /// @param limiter_in A NodeFilter_t function pointer that determines if the current
            /// node can be refined.
            /// \see AmrFcnTypes.h
            /// @author WVN
            void SetRefineLimiter(NodeFilter_t* limiter_in);
            
            /// @brief Returns true if this node and the given node have the same box dimensions
            /// @param node The node to check this node against
            /// @author WVN
            bool IsSameDimensionsAs(RefinementTreeNode* node);
            
            /// @brief Returns the level that this node is at.
            /// @author WVN
            int GetLevel(void);
            
            /// @brief Returns true if this node shares the edge indicated by edgeIndex.
            /// The edgeIndex values correspond to: (1, 2, 3, 4, [5, 6]) = (xmin, xmax, ymin, ymax, [zmin, zmax])
            /// @param edgeIndex The index of the edge in question
            /// @author WVN
            bool SharesEdgeWithHost(int edgeIndex);
            
            /// @brief Returns an array of size 2*CMF_DIM representing the bounding box of the current node.
            /// Returned as (xmin, xmax, ymin, ymax, [zmin, zmax])
            /// @author WVN
            std::vector<double> GetBlockBounds(void);
            
            /// @brief Returns an array of size 2*CMF_DIM representing the bounding box of the current node.
            /// Returned as (xmin, xmax, ymin, ymax, zmin=0, zmax=1)
            /// @author WVN
            Vec<double, 6> GetBlockBoundsVec6(void);
            
            /// @brief Recursively counts the number of terminal nodes and increments totalNumBlocks accordingly.
            /// Deprecated, should not be used.
            /// @param totalNumBlocks Pointer to a counter to be incremented
            /// @author WVN
            void RecursiveCountTerminal(int* totalNumBlocks);
            
            /// @brief Recursively counts the number of terminal nodes and increments totalNumBlocks accordingly.
            /// Deprecated, should not be used.
            /// @param totalNumBlocks Pointer to a counter to be incremented
            /// @param filter A function pointer that, if returns false on any given node, skips that node
            /// @author WVN
            void RecursiveCountTerminal(int* totalNumBlocks, NodeFilter_t filter);
            
            /// @brief Recursively writes points to the provided VtkBuffers objects. Calls WriteBlockDataToVtkBuffers
            /// with the same parameters.
            /// @param points VtkBuffer object representing the point buffer. Point coordinates are streamed into this
            /// @param edges VtkBuffer object representing the adjacency buffer. Point indices are streamed into this
            /// @param cellTypes VtkBuffer object representing the point buffer. Cell type IDs are streamed into this
            /// @param num A pointer to a running total
            /// @param filter A function pointer that, if returns false on any given node, skips that node
            /// \see VtkFile VtkBuffer VtkCellType WriteBlockDataToVtkBuffers
            /// @author WVN
            void RecursiveWritePointsToVtk(VtkBuffer& points, VtkBuffer& edges, VtkBuffer& cellTypes, int* num, NodeFilter_t filter);
            
            /// @brief Recursively writes points to the provided VtkBuffers objects. Calls WriteBlockDataToVtkBuffers
            /// with the same parameters.
            /// @param points VtkBuffer object representing the point buffer. Point coordinates are streamed into this
            /// @param edges VtkBuffer object representing the adjacency buffer. Point indices are streamed into this
            /// @param cellTypes VtkBuffer object representing the point buffer. Cell type IDs are streamed into this
            /// @param num A pointer to a running total
            /// \see VtkFile VtkBuffer VtkCellType WriteBlockDataToVtkBuffers
            /// @author WVN
            void RecursiveWritePointsToVtk(VtkBuffer& points, VtkBuffer& edges, VtkBuffer& cellTypes, int* num);
            
            /// @brief Writes node data to the provided VtkBuffers
            /// @param points VtkBuffer object representing the point buffer. Point coordinates are streamed into this
            /// @param edges VtkBuffer object representing the adjacency buffer. Point indices are streamed into this
            /// @param cellTypes VtkBuffer object representing the point buffer. Cell type IDs are streamed into this
            /// @param num Accumulates a running total of the number of blocks written. Generally can be passed a dummy variable.
            /// \see VtkFile VtkBuffer VtkCellType
            /// @author WVN
            void WriteBlockDataToVtkBuffers(VtkBuffer& points, VtkBuffer& edges, VtkBuffer& cellTypes, int* num);
            
            /// @brief Returns the number of neighbors of the current node.
            /// @author WVN
            int NumberOfNeighbors(void);
            
            /// @brief Returns the maximum length of the bounding box of the current node.
            /// @author WVN
            double MaxEdgeLength(void);
            
            /// @brief Returns the list of neighbors
            /// @author WVN
            std::vector<std::pair<RefinementTreeNode*, NodeEdge>>& Neighbors(void);
            
            /// @brief Reads tree descriptor from the parallel file
            /// @param file The file to read from
            /// @author WVN
            void ReadFromFile(ParallelFile& file);
            
            /// @brief Writes the tree descriptor to the parallel file
            /// @param file The file to write to
            /// @author WVN
            void WriteToFile(ParallelFile& file);
            
            /// @brief Returns a vector containing directionLevels
            /// @author WVN
            Vec3<int> GetDirectionLevels(void);
            
            /// @brief Returns a vector representing the coordinates of the center of the block
            /// @author WVN
            Vec3<double> GetBlockCenter(void);
            
            /// @brief used for debugging only, to be removed at a later date. Prints information about the neighbors of this block
            /// @author WVN
            void PrintNeighbors(void);
            
            /// @brief Returns the parent does if there is one, and NULL if there is no parent
            /// @author WVN
            RefinementTreeNode* GetParent(void)
            {
                return host;
            }
            
        private:
            /// @brief Locks the current node from being modified by recursive calls.
            /// \see Unlock
            /// @author WVN
            void Lock(void);
            
            /// @brief Unlocks the current node from being modified by recursive calls.
            /// \see Lock
            /// @author WVN
            void Unlock(void);
            
            /// @brief Returns an integer that can be used to augment the hash of the base block.
            /// @author WVN
            int GetHashableValue(void);
            
            /// @brief Returns true if the current node is locked.
            /// \see Unlock Lock
            /// @author WVN
            bool NodeIsLocked(void);
            
            /// @brief Internal, recursive version of Refine(). \see Refine
            /// @param newRefinementType A char interpreted as a 3-vector of bits indicating refinement directions
            /// @param recursiveLevel A char interpreted as a 3-vector of bits indicating refinement directions
            /// @author WVN
            void RefineRecursive(char newRefinementType, int recursiveLevel);
            
            /// @brief Determines, based on the current RefinementConstraint, whether toBeRefined should be refined
            /// based on the NodeEdge relationship with newChildNode. If so, sets newRefTypeOut to the appropriate
            /// refine type.
            /// @param newChildNode The node that has just been refined
            /// @param toBeRefined The node that may or may not be refined
            /// @param relationship The relationship between newChildNode and toBeRefined
            /// @param newRefTypeOut The refinement type that toBeRefined should be refined according to
            /// @author WVN
            static bool RefineRequiredFromRelationship(RefinementTreeNode* newChildNode, RefinementTreeNode* toBeRefined, NodeEdge relationship, char* newRefTypeOut);
            
            /// @brief Populate isOnBoundary based on host node information
            /// @author WVN
            void InheritDomainBoundaryInfo(void);
            
            /// @brief Populates directionLevels based on host node information
            /// @author WVN
            void DefineDirectionLevels(void);
            
            /// @brief Returns the 1-dimensional index of a child node based on location (interpreted as a 3-vector of bits) and refinementType.
            /// @param location The octant vector, a char interpreted as a 3-vector of bits, as in GetOctant
            /// @param refinementType the refinement type in consideration
            /// \see GetOctant
            /// @author WVN
            int GetIndexFromOctantAndRefineType(char location, char refinementType);
            
            /// @brief Returns the number of subnodes generated upon a certain refinement type, e.g. 3 refines in x anx y, so 
            /// 4 children are created
            /// @param refinementType the refinement type in consideration
            /// @author WVN
            int NumberOfNewSubNodes(char refinementType);
            
            /// @brief Returns the basis permutation according to the numbering convention of refinementType. Should not
            /// be called except for advanced usage.
            /// @param refinementType the refinement type in consideration
            /// @author WVN
            int GetCoordBasis(char refinementType);
            
            /// @brief Returns the basis reverse permutation according to the numbering convention of refinementType. Should not
            /// be called except for advanced usage.
            /// @param refinementType the refinement type in consideration
            /// @author WVN
            int GetInvCoordBasis(char refinementType);
            
            /// @brief Computes the bounding box of the current node based on the orientation, refinement type, and host node bounding box.
            /// @param hostBounds The boudning box of the host node
            /// @param refineType_in The refinement type that produced the current node
            /// @param refineOrientation_in The refinement orientation of the current node
            /// @author WVN
            void DefineBounds(std::vector<double> hostBounds, char refineType_in, char refineOrientation_in);
            
            /// @brief Computes the bounding box of the current node using the exact number system based on the refinement levels and whether or not the current node shares 
            /// @author WVN
            void DefineExactPositions(void);
            
            /// @brief Updates the neighbors of the current node's neighbors and the current node's children after a refinement
            /// @author WVN
            void UpdateNeighborsAfterRefinement(void);
            
            /// @brief Updates the neighbors of the current node's neighbors and the current node's children after a refinement
            /// @param the node to check for neighbor relationships
            /// @author WVN
            void CreateNeighborRelatioships(RefinementTreeNode* node);
            
            /// @brief The refinement type that produced this node (see RefinementTreeNode constructor for interpretation)
            char refineType;
            
            /// @brief The refinement orientation of this node with respect to host (see RefinementTreeNode constructor for interpretation)
            char refineOrientation;
            
            /// @brief Set to false if this node has children, true otherwise
            bool isTerminal;
            
            /// @brief Set to false if this node has children, true otherwise
            bool deallocSubTrees;
            
            /// @brief If true, should prevent recursive neighbor flood calls from looping back to current node
            bool isLocked;
            
            /// @brief The refinement type the generates the child nodes of the current node
            char subNodeRefinementType;
            
            /// @brief The block bounds of the current node, blockBounds = (xmin, xmax, ymin, ymax, [zmin, zmax])
            std::vector<double> blockBounds;
            
            /// @brief The RefinementConstraint that determines, when this node is refined, which of its neighbors (and subsequently their neighbors, etc...) must be refines as well
            RefinementConstraint::RefinementConstraint constraint;
            
            /// @brief An array to store the pointers of child nodes
            RefinementTreeNode** subNodes;
            
            /// @brief A pointer to the host or parent node of this node
            RefinementTreeNode* host;
            
            /// @brief Exact bounds of this node (can be compared without worrying about floating-point precision)
            ExactAmrPosition<amrBitType> exactBounds [2*CMF_DIM];
            
            /// @brief The number of children this node has
            int numSubNodes;
            
            /// @brief The number of refinement levels that it took to arrive at this node
            int level;
            
            /// @brief The number of refinement levels that it took to arrive at this node, considering each Cartesian direction separately
            int directionLevels[CMF_DIM];
            
            /// @brief A table of neighboring nodes and their relationship properties
            std::vector<std::pair<RefinementTreeNode*, NodeEdge>> neighbors;
            
            /// @brief Element d is true if the current node lies on the d'th face of the rootBlock
            bool isOnBoundary[2*CMF_DIM];
            
            /// @brief A function pointer that, if returns true for this node, prevents any futher refinement
            NodeFilter_t* refineLimiter;
            
            /// @brief A pointer to the original RefinementBlock object
            RefinementBlock* rootBlock;
            
            /// @brief Deprecated
            int iteratorIndex;
            
            /// @brief A value that can be used to identify this node in a VTK output
            int nodeTag;
            
            friend class NeighborIterator;
            friend class BlockIterator;
            friend class RefinementBlock;
    };
}

#include "BlockFilters.h"

#endif
