#ifndef ICMF_MESH_H
#define ICMF_MESH_H
#include <string>
#include "ICmfInputObject.h"
#include "ICmfMeshArrayHandler.h"
#include "CmfScreen.h"
namespace cmf
{
    namespace MeshType
    {
        enum MeshType
        {
            Cartesian,
            Unstructured
        };
    }
    
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
        ICmfMeshInfo(PropTreeLib::PropertySection& inputTree)
            : ICmfInputObject(inputTree){}
    };

    class ICmfMesh
    {
        public:
            /// @brief Default input-info constructor for ICmfMesh
            /// @param input ICmfMeshInfo defining the current mesh
            /// @param meshType_in the type of the currnet mesh
            /// @author WVN
            ICmfMesh(ICmfMeshInfo input, MeshType::MeshType meshType_in);

            /// @brief Empty destructor
            /// @author WVN
            ~ICmfMesh(void);

            /// @brief Returns the array handler object for the given mesh
            /// @author WVN
            virtual ICmfMeshArrayHandler* GetArrayHandler(void);

            /// @brief Defines a variable on the given mesh
            /// @param name The name of the variable
            /// @author WVN
            virtual ICmfMeshArray& DefineVariable(std::string name)=0;

        protected:
            /// @brief Title of the mesh, used for debugging and output
            std::string title;
            
            /// @brief The type of the mesh
            MeshType::MeshType meshType;
    };
}

#endif
