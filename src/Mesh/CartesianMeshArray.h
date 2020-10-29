#ifndef CMF_CARTESIAN_MESH_ARRAY_H
#define CMF_CARTESIAN_MESH_ARRAY_H
#include <string>
#include "ArrayInfo.h"
#include "ICmfMeshArray.h"
namespace cmf
{
    /// @brief Defines a MeshArray object Cartesian grids
    /// @author WVN
    class CartesianMeshArray : public ICmfMeshArray
    {
        public:
            /// @brief Constructor
            /// @param name The name of the variable
            /// @author WVN
            CartesianMeshArray(ArrayInfo info);
            
            /// @brief Empty destructor
            /// @author WVN
            ~CartesianMeshArray(void);
            
            /// @brief Explcity releases resources used by the current object
            /// @author WVN
            void Destroy(void);
    };
}

#endif