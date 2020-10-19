#include "CartesianMesh.h"

namespace cmf
{
    CartesianMesh::CartesianMesh(std::string title_in)
    {
        title = title_in;
        localInput.SetAsSubtree(mainInput[title]);
        localInput["blockDim"].MapTo(&blockDim) = new PropTreeLib::Variables::PTLStaticIntegerArray(CMF_DIM, "Base block dimensions");
        localInput["blockBounds"].MapTo(&blockBounds) = new PropTreeLib::Variables::PTLStaticDoubleArray(2*CMF_DIM, "Base block bounds");
        localInput["refinementConstraintType"].MapTo((int*)&refinementConstraintType)
            = new PropTreeLib::Variables::PTLAutoEnum(RefinementConstraint::free, RefinementConstraintStr, "Determines how refinements are constrained");
        localInput.StrictParse();
        blocks = new RefinementBlock(blockDim, blockBounds, refinementConstraintType);
    }
    
    RefinementBlock* CartesianMesh::Blocks(void)
    {
        return blocks;
    }
    
    CartesianMesh::~CartesianMesh(void)
    {
        delete blocks;
    }
}