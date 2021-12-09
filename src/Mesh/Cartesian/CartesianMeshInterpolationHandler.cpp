#include "CartesianMeshInterpolationHandler.h"

namespace cmf
{
    CartesianMeshInterpolationHandler::CartesianMeshInterpolationHandler(CartesianMeshArray& source_in, CartesianMesh& destination_in)
    {
        source = &source_in;
        destination = &destination_in;
    }
}