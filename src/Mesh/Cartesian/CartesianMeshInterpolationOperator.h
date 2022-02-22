#ifndef CMF_CARTESIAN_MESH_INTERPOLATION_OPERATOR_H
#define CMF_CARTESIAN_MESH_INTERPOLATION_OPERATOR_H
#include "CartesianMesh.h"
#include "CartesianMeshArray.h"
namespace cmf
{
    /// @brief A class that interpolates between two arrays on separate meshes
    class CartesianMeshInterpolationOperator
    {
        public:
            /// @brief Constructor, creates an interpolation pattern between the two meshes
            /// @param source_in The mesh to be interpolated from
            /// @param destination_in The mesh to interpolate onto
            /// @author WVN
            CartesianMeshInterpolationOperator(CartesianMesh& source_in, CartesianMesh& destination_in);
            
            /// @brief Executes the interpolation operation on the provided meshes, where sourceArray resides
            /// on the source mesh and destinationArray resides on the destination mesh. An exception is
            /// thrown if the meshes are not conformal or if the arrays do not lie on the proper meshes.
            /// @param sourceArray The mesh to be interpolated from
            /// @param destinationArray The mesh to interpolate onto
            /// @author WVN
            void Interpolate(CartesianMeshArray& sourceArray, CartesianMeshArray& destinationArray);
        
        private:
            /// @brief The donor mesh
            CartesianMesh* source;
            
            /// @brief The recipient mesh
            CartesianMesh* destination;
            
    };
}

#endif
