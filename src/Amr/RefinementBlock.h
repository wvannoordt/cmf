#ifndef RefinementBlock_H
#define RefinementBlock_H

#include <string>
#include <vector>
#include "AmrFcnTypes.h"
#include "PTL.h"
#include "RefinementTreeNode.h"
#include "BlockIterator.h"
#include "RefinementConstraint.h"
#include "IBlockIterable.h"
#include "ICmfHashable.h"
#include "ParallelFile.h"
#include "Vec.h"

namespace cmf
{
    class RefinementTreeNode;
    class BlockIterator;
    class IPostRefinementCallback;
    /// @brief Class that represents a grid and tree structure of RefinementTreeNode objects. Essentially represents a block structure for 
    /// a computational mesh
	/// @author WVN
    class RefinementBlock : public IBlockIterable, public ICmfHashable
    {
        public:
            /// @brief Constructor for the RefinementBlock object.
            /// @param blockDim_in The number of subdivisions to initially divide the bounding box into
            /// @param blockBounds_in The bounding box for the domain
            /// @param constraint_in A string title that will be used to look for option in the section of the input file (the section has the same name)
            /// \see RefinementConstraint.h
            /// @param periodicRefinement_in A boolean array of size CMF_DIM indicating whether or not refinement constraints are applied periodically
            /// @author WVN
            RefinementBlock(std::vector<int> blockDim_in, std::vector<double> blockBounds_in, RefinementConstraint::RefinementConstraint constraint_in, std::vector<bool> periodicRefinement_in);
            
            /// @brief Destructor for the RefinementBlock object. Explicitly calls Destroy and destroys all contained RefinementTreeNode objects.
            /// @author WVN
            ~RefinementBlock(void);
            
            /// @brief Called when destructor is called. Destroys all contained RefinementTreeNode objects.
            /// @author WVN
            void Destroy(void);
            
            /// @brief Draws each node to the provided TikzObject. For debugging and figure generation only.
            /// \pre Should only be used in 2D.
            /// @author WVN
            void Render(TikzObject* picture);
            
            /// @brief Draws each node to the provided TikzObject. For debugging and figure generation only.
            /// \pre Should only be used in 2D.
            /// @param debugger A function pointer that provides extra drawing code, called for each node
            /// @author WVN
            void Render(TikzObject* picture, DebugTikzDraw_t debugger);
            
            /// @brief Refines every contained RefinementTreeNode. Generally should not be used.
            /// @param refinementType The type of refinement to apply to all nodes
            /// @author WVN
            void RefineAll(char refinementType);
            
            /// @brief Randomly refined nodes within the current block. Generally should not be used except for demonstration.
            /// @author WVN
            void RefineRandom();
            
            /// @brief Refines the node that contains the coordinates provided.
            /// @param coords The coordinates to refine at
            /// @param refinementType The type of refinement to apply to the relevant node, if found
            /// @author WVN
            void RefineAt(double coords[CMF_DIM], char refinementType);
            
            /// @brief Refines the node that contains the coordinates provided.
            /// @param coords The coordinates to refine at
            /// @param refinementType The type of refinement to apply to the relevant node, if found
            /// @author WVN
            void RefineAt(Vec3<double> coords, char refinementType);
            
            /// @brief Refines the list of the provided nodes and calls all post-refinement callback functions. This is the preferred way of refining large numbers of nodes.
            /// @param pairs A map containing a list of nodes and the types of refinement to apply
            /// @author WVN
            void RefineNodes(std::map<RefinementTreeNode*, char> pairs);
            
            /// @brief Refines the list of the provided nodes and calls all post-refinement callback functions. This is the preferred way of refining large numbers of nodes.
            /// @param nodes A list of nodes to refine
            /// @param refineType the refinement type to use
            /// @author WVN
            void RefineNodes(std::vector<RefinementTreeNode*>& nodes, char refineType);
            
            /// @brief Refines the list of the provided nodes and calls all post-refinement callback functions. This is the preferred way of refining large numbers of nodes.
            /// @param nodes A list of nodes to refine
            /// @param refineTypes the list of refinement types to use
            /// @author WVN
            void RefineNodes(std::vector<RefinementTreeNode*>& nodes, std::vector<char>& refineTypes);
            
            /// @brief Returns the node that contains the provided coordinates, or NULL if none are found
            /// @param coords The coordinates to retrieve at
            /// @author WVN
            RefinementTreeNode* GetNodeAt(double coords[CMF_DIM]);
            
            /// @brief Returns the node that contains the provided coordinates, or NULL if none are found
            /// @param coords The coordinates to retrieve at
            /// @author WVN
            RefinementTreeNode* GetNodeAt(Vec3<double>& coords);
            
            /// @brief Returns the node that contains the provided coordinates, or NULL if none are found
            /// @param x x-Coordinate
            /// @param y y-Coordinate
            /// @param z z-Coordinate
            /// @author WVN
            RefinementTreeNode* GetNodeAt(double x, double y, double z);
            
            /// @brief Returns the node that contains the provided coordinates, or NULL if none are found
            /// @param x x-Coordinate
            /// @param y y-Coordinate
            /// @author WVN
            RefinementTreeNode* GetNodeAt(double x, double y);
            
            /// @brief Returns true if the given coordinates lie within the domain boundaries.
            /// @param coords The coordinates to test
            /// @param idx The 2-D or 3-D indices of the resulting node
            /// @author WVN
            bool PointIsInDomain(double coords[CMF_DIM], int* idx);
            
            /// @brief Returns true if the given coordinates lie within the domain boundaries.
            /// @param coords The coordinates to test
            /// @author WVN
            bool PointIsInDomain(double coords[CMF_DIM]);
            
            /// @brief Sets a hard refinement limit on all nodes.
            /// @param limiter_in A NodeFilter_t function pointer that, if returns true on a given node, prevents it from being refined
            /// @author WVN
            void SetRefineLimitCriterion(NodeFilter_t limiter_in);
            
            /// @brief \see IBlockIterable::GetRefinementBlockObject
            /// @author WVN
            RefinementBlock* GetRefinementBlockObject(void);
            
            /// @brief Outputs a VTK file that contains only the node boxes. Used only for debugging/
            /// @param filename A ".vtk" filename to write to
            /// @author WVN
            void OutputDebugVtk(std::string filename);
            
            /// @brief Outputs a VTK file that contains only the node boxes. Used only for debugging/
            /// @param filename A ".vtk" filename to write to
            /// @param filter A function pointer that, if returns false on any given node, skips that node
            /// @author WVN
            void OutputDebugVtk(std::string filename, NodeFilter_t filter);
            
            /// @brief Called every time a new node is created via refinement, used to build enumerators.
            /// @param newNode A pointer to a node to register
            /// @author WVN
            void RegisterNewChildNode(RefinementTreeNode* newChild);
            
            /// @brief Called every time a node refined, used to build enumerators.
            /// @param newNode A pointer to a node to register
            /// @author WVN
            void RegisterNewParentNode(RefinementTreeNode* newParent);
            
            /// @brief Returns the total number of nodes that are contained within the current block
            /// @author WVN
            size_t Size(void);
            
            /// @brief Returns the total number of nodes that are contained within the current block
            /// @param filter A filter to apply to the counted blocks
            /// @author WVN
            size_t Size(NodeFilter_t filter);
            
            /// @brief Returns a pointer to a list of all nodes
            /// @author WVN
            std::vector<RefinementTreeNode*>* GetAllNodes(void) override;
            
            /// @brief Sets currentIterator to the provided value
            /// @param iter A pointer to the iterator to set
            /// @author WVN
            void SetIterator(BlockIterator* iter);
            
            /// @brief Returns an array of size 2*CMF_DIM representing the bounding box of the current refinement box.
            /// Returned as (xmin, xmax, ymin, ymax, [zmin, zmax])
            /// @author WVN
            std::vector<double> GetBlockBounds(void);
            
            /// @brief Outputs a file with neighbor information (DEBUGGING ONLY!)
            /// @author WVN
            /// @param filename The file to output
            void OutputNeighborsToFile(std::string filename);
            
            /// @brief Outputs a list of nodes to a vtk file
            /// @param nodeList A list of nodes to be output
            /// @param filename The name of the file to be output, should have a ".vtk" file extension
            /// @author WVN
            static void OutputNodesToVtk(const std::vector<RefinementTreeNode*>& nodeList, std::string filename);
            
            /// @brief Prevents an error from being thrown when HandleRefinementQueryOutsideDomain is called \see HandleRefinementQueryOutsideDomain
            /// @author WVN
            void NoCrashOnQueryOutsideDomain(void);
            
            /// @brief Calls all post-refinement callback functions in postRefinementCallbackObjects
            /// @author WVN
            void PostRefinementCallbacks(void);
            
            /// @brief Call all pre-refinement callbacks
            /// @param toBeRefined List of nodes that are about to be refined
            /// @author WVN
            void PreRefinementCallbacks(const std::vector<cmf::RefinementTreeNode*>& toBeRefined) const;
            
            /// @brief Adds a IPostRefinementCallback object to the post-refinement call list. Returns the
            /// index/position of the added object in the list
            /// @param obj The object to add
            /// @author WVN
            int AddPostRefinementCallbackObject(IPostRefinementCallback* obj);
            
            /// @brief Returns the refinement constraint type
            /// @author WVN
            RefinementConstraint::RefinementConstraint GetRefinementConstraintType(void);
            
            /// @brief Reads tree descriptor from the parallel file
            /// @param file The file to read from
            /// @author WVN
            void ReadFromFile(ParallelFile& file);
            
            /// @brief Writes the tree descriptor to the parallel file
            /// @param file The file to write to
            /// @author WVN
            void WriteToFile(ParallelFile& file);
            
            /// @brief Returns the maximum grid level, by direction
            /// @author WVN
            Vec3<int> GetMaxLevel(void)
            {
                return maxLevel;
            }
            
        private:
            
            /// @brief Creates the initial grid of RefinementTreeNode objects
            /// @author WVN
            void DefineTrunks(void);
            
            /// @brief Handles the situation where a refinement query lands outside the refinement block. Currently does nothing.
            /// @param coords the coordinates of the query
            /// @author WVN
            void HandleRefinementQueryOutsideDomain(double coords[CMF_DIM]);
            
            /// @brief An array of size CMF_DIM that represents the initial block dimensions
            std::vector<int> blockDim;
            
            /// @brief The product of the elements of blockDim
            int totalNumTrunks;
            
            /// @brief An array of size 2*CMF_DIM the represents the block boundaries of the current refinement block as (xmin, xmax, ymin, ymax, [zmin, zmax])
            std::vector<double> blockBounds;
            
            /// @brief The maximum level (by direction) of any node in this collection
            Vec3<int> maxLevel;
            
            /// @brief An array of size CMF_DIM that represent the size of a single initial node in each dimension
            double dx[CMF_DIM];
            
            ///@brief A boolean array of size CMF_DIM indicating whether or not refinement constraints are applied periodically
            std::vector<bool> periodicRefinement;
            
            /// @brief An array to store the initial nodes of this refinement block
            RefinementTreeNode** trunks;
            
            /// @brief Set to true when trunks array is allocated
            bool deallocTrunks;
            
            /// @brief If true, coordinate queries outside domain will not cause crashes
            bool crashOnQueryOutsideDomain;
            
            /// @brief See RefinementTreeNode::RefinementConstraint. Applied to all contained nodes
            RefinementConstraint::RefinementConstraint refinementConstraintType;
            
            /// @brief See RefinementTreeNode::refineLimiter. Applied to all contained nodes
            NodeFilter_t refineLimiter;
            
            /// @brief A list of pointers to all nodes contained within this refinement block
            std::vector<RefinementTreeNode*> allNodes;
            
            /// @brief A list of recently-created child nodes that are buffered for processing by post-refinement callbacks
            std::vector<RefinementTreeNode*> newChildNodes;
            
            /// @brief A list of recently-refined parents nodes that are buffered for processing by post-refinement callbacks
            std::vector<RefinementTreeNode*> newParentNodes;
            
            /// @brief A list of objects that have refinement callbacks to be processed after refinements
            std::vector<IPostRefinementCallback*> postRefinementCallbackObjects;
            
            friend class RefinementTreeNode;
            friend class BlockIterator;
    };
}

#endif
