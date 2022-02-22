#include "CartesianMeshInterpolationOperator.h"
#include "StringUtils.h"
#include "CmfScreen.h"
namespace cmf
{
    CartesianMeshInterpolationOperator::CartesianMeshInterpolationOperator(CartesianMesh& source_in, CartesianMesh& destination_in)
    {
        source = &source_in;
        destination = &destination_in;
    }
    
    void CartesianMeshInterpolationOperator::Interpolate(CartesianMeshArray& sourceArray, CartesianMeshArray& destinationArray)
    {
        std::string frmt = "Interpolating variable \"{}\" to variable \"{}\"";
        WriteLine(3, strformat(frmt, sourceArray.GetFullName(), destinationArray.GetFullName()));
        
    }
}