#ifndef RefinementBlock_H
#define RefinementBlock_H

#include <string>
#include <vector>
#include "AmrFcnTypes.h"
#include "PropTreeLib.h"
#include "RefinementTreeNode.h"
#include "BlockIterator.h"
#include "RefinementConstraint.h"

namespace cmf
{
    class RefinementTreeNode;
    /// @brief Class that represents a grid and tree structure of RefinementTreeNode objects. Essentially represents a block structure for 
    /// a computational mesh
	/// @author WVN
    class RefinementBlock
    {
        public:
            /// @brief Constructor for the RefinementBlock object.
            /// @param title A string title that will be used to look for option in the section of the input file (the section has the same name)
            /// @author WVN
            RefinementBlock(std::string title);
            
            /// @brief Destructor for the RefinementBlock object. Explicitly calls Destroy and destroys all contained RefinementTreeNode objects.
            /// @author WVN
            ~RefinementBlock(void);
            
            /// @brief Deprecated
            /// @author WVN
            void Print(void);
            
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
            
            /// @brief Returns the node that contains the provided coordinates, or NULL if none are found
            /// @param coords The coordinates to retrieve at
            /// @author WVN
            RefinementTreeNode* GetNodeAt(double coords[CMF_DIM]);
            
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
            
            /// @brief Outputs a VTK file that contains only the node boxes. Used only for debugging/
            /// @param filename A ".vtk" filename to write to
            /// @author WVN
            void OutputDebugVtk(std::string filename);
            
            /// @brief Called every time a new node is created via refinement, used to build enumerators.
            /// @param newNode A pointer to a node to register
            /// @author WVN
            void RegisterNewNode(RefinementTreeNode* newNode);
            
            /// @brief Returns the total number of nodes that are contained within the current block
            /// @author WVN
            size_t Size(void);
        private:
            
            /// @brief Creates the initial grid of RefinementTreeNode objects
            /// @author WVN
            void DefineTrunks(void);
            
            /// @brief Handles the situation where a refinement query lands outside the refinement block. Currently does nothing.
            /// @param coords the coordinates of the query
            /// @author WVN
            void HandleRefinementQueryOutsideDomain(double coords[CMF_DIM]);
            
            /// @brief PropertyTree object that contains the input data for this object (see <a href="https://github.com/wvannoordt/PropTreeLib">PropTreeLib</a>)
            PropTreeLib::PropertyTree localInput;
            
            /// @brief An array of size CMF_DIM that represents the initial block dimensions
            int* blockDim;
            
            /// @brief The product of the elements of blockDim
            int totalNumTrunks;
            
            /// @brief An array of size 2*CMF_DIM the represents the block boundaries of the current refinement block as (xmin, xmax, ymin, ymax, [zmin, zmax])
            double* blockBounds;
            
            /// @brief An array of size CMF_DIM that represent the size of a single initial node in each dimension
            double dx[CMF_DIM];
            
            /// @brief An array to store the initial nodes of this refinement block
            RefinementTreeNode** trunks;
            
            /// @brief Set to true when trunks array is allocated
            bool deallocTrunks;
            
            /// @brief See RefinementTreeNode::RefinementConstraint. Applied to all contained nodes
            RefinementConstraint::RefinementConstraint constraint;
            
            /// @brief See RefinementTreeNode::refineLimiter. Applied to all contained nodes
            NodeFilter_t refineLimiter;
            
            /// @brief A list of pointers to all nodes contained within this refinement block
            std::vector<RefinementTreeNode*> allNodes;
        friend class RefinementTreeNode;
        friend class BlockIterator;
    };
}

#endif
