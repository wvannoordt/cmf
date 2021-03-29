#ifndef ICMF_ARRAY_HANDLER_H
#define ICMF_ARRAY_HANDLER_H
#include "ArrayInfo.h"
#include "CmfScreen.h"
#include "ICmfMeshArray.h"
#include <map>
#include "ICmfDataBaseReadWriteObject.h"
namespace cmf
{
    class ICmfMesh;
    /// @brief Defines a general MeshArrayHandler object for various grid types
    /// @author WVN
    class ICmfMeshArrayHandler : public ICmfDataBaseReadWriteObject
    {
        public:
            /// @brief Empty constructor
            /// @author WVN
            ICmfMeshArrayHandler(void);

            /// @brief Empty destructor
            /// @author WVN
            ~ICmfMeshArrayHandler(void);

            /// @brief Creates a new variable with the given name
            /// @param info The constructor information for the current array
            /// @author WVN
            virtual ICmfMeshArray* CreateNewVariable(ArrayInfo info);
            
            /// @brief Returns the variable with the given name
            /// @param name The name of the variable to fetch
            /// @author WVN
            virtual ICmfMeshArray* GetVariable(std::string name)=0;
            
            /// @brief Returns a string that uniquely identifies the current object in a database
            /// @author WVN
            virtual std::string DataBaseName(void) override final;
            
            /// @brief Adds the set of prerequisite objects to objectsRequiredBeforeAddingToDataBase
            ///@author WVN
            virtual void SetRequiredPrereqtuisiteDataBaseObjects(void) override final;
            
            /// @brief Adds the set of automatically added objects to objectsToAutomaticallyAddWhenAddingToDataBase
            ///@author WVN
            virtual void SetAutomaticallyAddedObjects(void) override final;

            /// @brief Checks if a variable with the given name exists
            /// @param name The name to check
            /// @author WVN
            bool VariableExists(std::string name);

            /// @brief Explcity releases resources used by the current object
            /// @author WVN
            void Destroy(void);
            
            /// @brief Returns the mesh that the handler is defined over
            /// @author WVN
            ICmfMesh* Mesh(void);

        protected:
            /// @brief A map from variable names to the underlying objects
            std::map<std::string, ICmfMeshArray*> varList;
            
            /// @brief The underlying mesh
            ICmfMesh* baseMesh;
    };
}

#endif
