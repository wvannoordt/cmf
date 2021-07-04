#ifndef ICMF_MESH_H
#define ICMF_MESH_H
#include <string>
#include "ICmfInputObject.h"
#include "ICmfMeshArrayHandler.h"
#include "CmfScreen.h"
#include "ParallelGroup.h"
#include "ICmfDataBaseReadWriteObject.h"
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
        ICmfMeshInfo(PTL::PropertySection& inputTree)
            : ICmfInputObject(inputTree){}
            
        /// @brief Constructor for manual definition
        ICmfMeshInfo(void) {}
    };

    class ICmfMesh: public ICmfDataBaseReadWriteObject
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
            
            /// @brief Returns the parallel group responsible for this mesh
            /// @author WVN
            ParallelGroup* GetGroup(void) {return meshGroup;}
            
            /// @brief Indicates whether the object has a ParallelGroup associated with it. Default implementation returns true for safety.
            /// If this returns false, it is assumed that the object is treated in parallel
            ///@author WVN
            virtual bool HasParallelGroup(void) override final {return true;}
            
            /// @brief Returns the parallel group for the object, or NULL if it is a serial object
            /// @author WVN
            virtual ParallelGroup* GetDatabaseParallelGroup(void) override final {return this->GetGroup();}

            /// @brief Returns the array handler object for the given mesh
            /// @author WVN
            virtual ICmfMeshArrayHandler* GetArrayHandler(void)=0;

            /// @brief Defines a variable on the given mesh
            /// @param name The name of the variable
            /// @author WVN
            virtual ICmfMeshArray& DefineVariable(std::string name)=0;
            
            /// @brief Returns the mesh title
            /// @author WVN
            std::string GetTitle(void);
            
            /// @author WVN
            /// @brief Returns a mesh array with the given name
            /// @param name The name of the array to fetch
            virtual ICmfMeshArray& operator [] (std::string name)=0;

        protected:
            /// @brief Title of the mesh, used for debugging and output
            std::string title;
            
            /// @brief The type of the mesh
            MeshType::MeshType meshType;
            
            /// @brief The parallel group that processes this mesh
            ParallelGroup* meshGroup;
            
            /// @brief The handler object for this mesh
            ICmfMeshArrayHandler* baseMeshArrayHandler = NULL;
    };
}

#endif
