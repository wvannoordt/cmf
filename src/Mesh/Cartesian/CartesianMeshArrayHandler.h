#ifndef CMF_CARTESIAN_MESH_ARRAY_HANDLER
#define CMF_CARTESIAN_MESH_ARRAY_HANDLER
#include "ICmfMeshArrayHandler.h"
#include "CartesianMeshArray.h"
#include "IPostRefinementCallback.h"
#include "AmrFcnTypes.h"
#include "CartesianMeshExchangeHandler.h"
#include <string>
#include <map>
namespace cmf
{
    class CartesianMesh;
    /// @brief Defines a MeshArrayHandler object for Cartesian meshes
    /// @author WVN
    class CartesianMeshArrayHandler : public ICmfMeshArrayHandler, public IPostRefinementCallback
    {
        friend class CartesianMeshArray;
        public:
            /// @brief Constructor
            /// @author WVN
            CartesianMeshArrayHandler(CartesianMesh* mesh_in);
            
            /// @brief Destructor
            /// @author WVN
            ~CartesianMeshArrayHandler(void);
            
            /// @brief The callback function for new nodes
            /// @param newNodes The newly refined nodes to be handled
            /// @author WVN
            void OnPostRefinementCallback(std::vector<RefinementTreeNode*>& newNodes);
            
            /// @brief Creates a new variable with the given name
            /// @param info Information about the variable \see ArrayInfo
            /// @param filter A block filter defining the domain of the variable
            /// @author WVN
            CartesianMeshArray* CreateNewVariable(ArrayInfo info, NodeFilter_t filter);
            
            /// @brief Retrieves a variable with the given name
            /// @param name The name of the variable to fetch
            /// @author WVN
            CartesianMeshArray* GetVariable(std::string name);
            
            /// @brief Checks if a variable with the given name exists
            /// @param name The name to check
            /// @author WVN
            bool VariableExists(std::string name);
            
            /// @brief Creates the CartesianMeshExchangeHandler object for the current array handler.
            /// @param inputInfo A constructor object to create the CartesianMeshExchangeHandler
            /// @author WVN
            void CreateExchangeHandler(CartesianMeshExchangeInfo& inputInfo);
            
            /// @brief Returns the default exchange handler for this mesh
            /// @author WVN
            CartesianMeshExchangeHandler* GetDefaultExchangeHandler(void);
        
        private:
            
            /// @brief The underlying mesh object
            CartesianMesh* mesh;
            
            /// @brief The object responsible for handling mesh exchanges.
            /// \pre Note that this is not built in the constructor, 
            /// but is created from an external call from CartesianMesh::CartesianMesh
            CartesianMeshExchangeHandler* defaultHandler;
            
            /// @brief Indicates whether or not the current object should delete defaultHandler
            bool requireDeleteDefaultHandler;
        
            template <typename arType, const int elementRank> friend struct BlockArray;
    };
}

#endif