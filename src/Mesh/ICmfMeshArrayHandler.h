#ifndef ICMF_ARRAY_HANDLER_H
#define ICMF_ARRAY_HANDLER_H
#include "ArrayInfo.h"
#include "CmfScreen.h"
#include "ICmfMeshArray.h"
#include <map>
namespace cmf
{
    class ICmfMesh;
    /// @brief Defines a general MeshArrayHandler object for various grid types
    /// @author WVN
    class ICmfMeshArrayHandler
    {
        public:
            /// @brief Empty constructor
            /// @author WVN
            ICmfMeshArrayHandler(void);

            /// @brief Empty destructor
            /// @author WVN
            ~ICmfMeshArrayHandler(void);

            /// @brief Creates a new variable with the given name
            /// @author WVN
            virtual void CreateNewVariable(ArrayInfo info);

            /// @brief Checks if a variable with the given name exists
            /// @param name The name to check
            /// @author WVN
            bool VariableExists(std::string name);

            /// @brief Explcity releases resources used by the current object
            /// @author WVN
            void Destroy(void);

        protected:
            /// @brief A map from variable names to the underlying objects
            std::map<std::string, ICmfMeshArray*> varList;
            
            /// @brief The underlying mesh
            ICmfMesh* mesh;
    };
}

#endif
