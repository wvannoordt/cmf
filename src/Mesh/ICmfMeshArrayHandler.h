#ifndef ICMF_ARRAY_HANDLER_H
#define ICMF_ARRAY_HANDLER_H
#include "ArrayInfo.h"
#include "CmfScreen.h"
#include "ICmfMeshArray.h"
#include "ICmfMesh.h"
#include <map>
namespace cmf
{
    /// @brief Defines a general MeshArrayHandler object for various grid types
    /// @author WVN
    class ICmfMeshArrayHandler
    {
        public:
            /// @brief Empty constructor
            /// @author WVN
            ICmfMeshArrayHandler(void){}
            
            /// @brief Empty destructor
            /// @author WVN
            ~ICmfMeshArrayHandler(void){Destroy();}
            
            /// @brief Creates a new variable with the given name
            /// @author WVN
            virtual void CreateNewVariable(ArrayInfo info){}
            
            /// @brief Checks if a variable with the given name exists
            /// @param name The name to check
            /// @author WVN
            bool VariableExists(std::string name){return (varList.find(name)!=varList.end());}
            
            /// @brief Explcity releases resources used by the current object
            /// @author WVN
            void Destroy(void)
            {
                for (std::map<std::string, ICmfMeshArray*>::iterator it = varList.begin(); it != varList.end(); it++)
                {
                    WriteLine(6, "Free var \"" + it->first + "\"");
                    it->second->Destroy();
                    delete it->second;
                }
            }
        
        protected:
            /// @brief A map from variable names to the underlying objects
            std::map<std::string, ICmfMeshArray*> varList;
            
            /// @brief The mesh over which the variable handler is defined
            ICmfMesh* mesh;
    };
}

#endif