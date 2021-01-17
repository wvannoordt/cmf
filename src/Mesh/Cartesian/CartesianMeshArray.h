#ifndef CMF_CARTESIAN_MESH_ARRAY_H
#define CMF_CARTESIAN_MESH_ARRAY_H
#include <string>
#include <map>
#include "ArrayInfo.h"
#include "ICmfMeshArray.h"
#include "AmrFcnTypes.h"
#include "IBlockIterable.h"
#include "BlockIterator.h"
#include "DataExchangePattern.h"
namespace cmf
{
    class CartesianMeshArrayHandler;
    /// @brief Defines a MeshArray object Cartesian grids
    /// @author WVN
    class CartesianMeshArray : public ICmfMeshArray, public IBlockIterable
    {
        public:
            
            using ICmfMeshArray::operator=;
            
            /// @brief Constructor
            /// @param name The name of the variable
            /// @author WVN
            CartesianMeshArray(ArrayInfo info, CartesianMeshArrayHandler* handler_in, NodeFilter_t filter_in);
            
            /// @brief Empty destructor
            /// @author WVN
            ~CartesianMeshArray(void);
            
            /// @brief Explcity releases resources used by the current object
            /// @author WVN
            void Destroy(void);
            
            /// @brief Returns the size of the total data stored in a block
            /// @author WVN
            size_t GetArraySizePerBlock(void);
            
            /// @brief Returns the blocks to be iterated over
            /// @author WVN
            std::vector<RefinementTreeNode*>* GetAllNodes(void);
            
            /// @brief Returns the total number of nodes that are contained within the iterable object
            /// @author WVN
            size_t Size(void);
            
            /// @brief Checks if a given block has a definition for the current variable
            /// @param node The block to check
            /// @author WVN
            bool IsSupportedBlock(RefinementTreeNode* node);
            
            /// @brief Populates pointerMap with the appropriate block-to-pointer mapping
            /// @author WVN
            void DefinePointerMap(void);
            
            /// @brief Returns the name of this variable including the name of the mesh
            /// @author WVN
            std::string GetFullName(void);
        
            /// @brief Allows for direct indexing using a block iterator
            /// @param it The block iterator to index with
            /// @author WVN
            void* operator [] (BlockIterator& it);
            
            /// @brief \see IBlockIterable::GetRefinementBlockObject
            /// @author WVN
            RefinementBlock* GetRefinementBlockObject(void);
            
            /// @brief Performs data exchanges to and from neighboring blocks
            /// @author WVN
            void Exchange(void);
            
        private:
            
            /// @brief Allocates the unferlying pointer
            /// @author WVN
            void Allocate(void);
            
            /// @brief Populates definedNodes
            /// @author WVN
            void GetDefinedNodes(void);
            
            /// @brief Creates the exchange pattern for this array
            /// @author WVN
            void CreateExchangePattern(void);
            
            /// @brief A filter definig the blocks that this variable is defined over
            NodeFilter_t filter;
            
            /// @brief The handler responsible for this array
            CartesianMeshArrayHandler* handler;
            
            /// @brief A map that holds the pointers for each block
            std::map<RefinementTreeNode*, void*> pointerMap;
            
            /// @brief Tells whether or not the underlying pointer is allocated or not
            bool isAllocated;
            
            /// @brief The parallel exchange pattern for this mesh array
            DataExchangePattern* exchangePattern;
            
            /// @brief Nodes over which this variable is defined
            std::vector<RefinementTreeNode*> definedNodes;
    };
}

#endif