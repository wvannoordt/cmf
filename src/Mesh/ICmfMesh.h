#ifndef ICMF_MESH_H
#define ICMF_MESH_H
#include <string>
#include "ICmfInputObject.h"
namespace cmf
{
    /// @brief Defines a general input structs for a mesh
    /// @author WVN
    struct ICmfMeshInfo : ICmfInputObject
    {
        /// @brief temporary dummy variable
        double dummyVariable;
        
        /// @brief Constructor for the ICmfMeshInfo object.
        /// @param title_in title of the mesh
        /// @param inputTree PropertryTree to be read from
        /// @author WVN
        ICmfMeshInfo(std::string title_in, PropTreeLib::PropertyTree& inputTree)
            : ICmfInputObject(title_in, inputTree){}
    };
    
    class ICmfMesh
    {
        public:
            ICmfMesh(ICmfMeshInfo input){title = input.title;}
            
        protected:
            /// @brief Title of the mesh, used for debugging and output
            std::string title;
    };
}

#endif