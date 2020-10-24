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
#include "ICmfMesh.h"
#include "cmf.h"

namespace cmf
{
    /// @brief A struct containing all the input information for a Cartesian mesh
    /// @author WVN
    struct CartesianMeshInputInfo : ICmfMeshInfo
    {
        /// @brief An array of size CMF_DIM that represents the initial block dimensions
        int* blockDim;
        
        /// @brief An array of size 2*CMF_DIM the represents the block boundaries of the current refinement block as (xmin, xmax, ymin, ymax, [zmin, zmax])
        double* blockBounds;
        
        /// @brief See RefinementTreeNode::RefinementConstraint. Applied to all contained nodes
        RefinementConstraint::RefinementConstraint refinementConstraintType;
        
        /// @brief Constructor for the CartesianMeshInputInfo object.
        /// @param title_in title of the mesh
        /// @param inputTree PropertryTree to be read from
        /// @author WVN
        CartesianMeshInputInfo(std::string title_in, PropTreeLib::PropertyTree& inputTree) : ICmfMeshInfo(title_in, inputTree)
        {
            Define();
            Parse();
        }
        
        void Define()
        {
            objectInput["blockDim"].MapTo(&blockDim) = new PropTreeLib::Variables::PTLStaticIntegerArray(CMF_DIM, "Base block dimensions");
            objectInput["blockBounds"].MapTo(&blockBounds) = new PropTreeLib::Variables::PTLStaticDoubleArray(2*CMF_DIM, "Base block bounds");
            objectInput["refinementConstraintType"].MapTo((int*)&refinementConstraintType)
                = new PropTreeLib::Variables::PTLAutoEnum(RefinementConstraint::free, RefinementConstraintStr, "Determines how refinements are constrained");
        }
    };
    class CartesianMesh : ICmfMesh
    {
        public:
            /// @brief Constructor for the CartesianMesh object.
            /// @param input Input data for the 
            /// @author WVN
            CartesianMesh(CartesianMeshInputInfo input);
            
            /// @brief Destructor for the CartesianMesh object.
            /// @author WVN
            ~CartesianMesh(void);
            
            /// @brief Returns the underlying RefinementBlock object
            /// @author WVN
            RefinementBlock* Blocks(void);
        private:
            
            /// @brief contains the mesh blocks of the current mesh
            RefinementBlock* blocks;
            
            /// @brief An array of size CMF_DIM that represents the initial block dimensions
            int* blockDim;
            
            /// @brief An array of size 2*CMF_DIM the represents the block boundaries of the current refinement block as (xmin, xmax, ymin, ymax, [zmin, zmax])
            double* blockBounds;
            
            /// @brief See RefinementTreeNode::RefinementConstraint. Applied to all contained nodes
            RefinementConstraint::RefinementConstraint refinementConstraintType;
    };
}

#endif