#ifndef CMF_CARTESIAN_MESH_HANDLER_H
#define CMF_CARTESIAN_MESH_HANDLER_H
#include "CartesianMesh.h"
#include "CartesianMeshArray.h"
namespace cmf
{
    /// @brief A helper class containing interpolation algorithms for placing an array on a
    /// Cartesian mesh that does not contain it
    class CartesianMeshInterpolationHandler
    {
        public:
            /// @brief Constructor
            /// @param source_in The array to be interpolated
            /// @param destination_in The mesh to interpolate onto
            /// @author WVN
            CartesianMeshInterpolationHandler(CartesianMeshArray& source_in, CartesianMesh& destination_in);
        
        private:
            
            /// @brief The array to be interpolated
            CartesianMeshArray* source;
            
            /// @brief The mesh to interpolate onto
            CartesianMesh* destination;
            
    };
}

#endif
