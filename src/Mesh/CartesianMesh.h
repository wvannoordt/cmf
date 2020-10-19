#ifndef CMF_CARTESIAN_MESH_H
#define CMF_CARTESIAN_MESH_H

#include <string>
#include <vector>
#include "AmrFcnTypes.h"
#include "PropTreeLib.h"
#include "RefinementTreeNode.h"
#include "BlockIterator.h"
#include "RefinementConstraint.h"
#include "RefinementBlock.h"
#include "cmf.h"

namespace cmf
{
    class CartesianMesh
    {
        public:
            /// @brief Constructor for the CartesianMesh object.
            /// @param title_in A name to give this domain
            /// @author WVN
            CartesianMesh(std::string title_in);
            
            /// @brief Destructor for the CartesianMesh object.
            /// @author WVN
            ~CartesianMesh(void);
            
            /// @brief Returns the underlying RefinementBlock object
            /// @author WVN
            RefinementBlock* Blocks(void);
        private:
            
            /// @brief contains the mesh blocks of the current mesh
            RefinementBlock* blocks;
            
            /// @brief PropertyTree object that contains the input data for this object (see <a href="https://github.com/wvannoordt/PropTreeLib">PropTreeLib</a>)
            PropTreeLib::PropertyTree localInput;
            
            /// @brief The title of the current mesh
            std::string title;
            
            /// @brief An array of size CMF_DIM that represents the initial block dimensions
            int* blockDim;
            
            /// @brief An array of size 2*CMF_DIM the represents the block boundaries of the current refinement block as (xmin, xmax, ymin, ymax, [zmin, zmax])
            double* blockBounds;
            
            /// @brief See RefinementTreeNode::RefinementConstraint. Applied to all contained nodes
            RefinementConstraint::RefinementConstraint refinementConstraintType;
    };
}

#endif