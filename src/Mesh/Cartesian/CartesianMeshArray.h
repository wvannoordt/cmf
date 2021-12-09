#ifndef CMF_CARTESIAN_MESH_ARRAY_H
#define CMF_CARTESIAN_MESH_ARRAY_H
#include <string>
#include <map>
#include <initializer_list>
#include <memory>

#include "ArrayInfo.h"
#include "ICmfMeshArray.h"
#include "AmrFcnTypes.h"
#include "IBlockIterable.h"
#include "BlockIterator.h"
#include "DataExchangePattern.h"
#include "Typedefs.h"
#include "BlockInfo.h"
#include "CartesianMeshBuffer.h"
#include "IPostRefinementCallback.h"
namespace cmf
{
    class CartesianMeshArray;
    
    /// @brief Represents a pairing of a CartesianMeshArray with a data pointer for a block
    /// @author WVN
    struct CartesianMeshArrayPointerPair
    {
        /// @brief The array that the pointer lies on a block of
        CartesianMeshArray* array;
        
        /// @brief The data pointer
        void* pointer;
        
        /// @brief Indicates whether or not an error is thrown
        bool enforceElementSizeMatches;
    };
    
    class CartesianMeshArrayHandler;
    class CartesianMesh;
    /// @brief Defines a MeshArray object Cartesian grids
    /// @author WVN
    class CartesianMeshArray : public ICmfMeshArray, public IBlockIterable, public IPostRefinementCallback
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
            
            /// @brief Returns the size of the total data stored in a block (elements, not bytes)
            /// @author WVN
            size_t GetArraySizePerBlock(void);
            
            /// @brief Returns the size of the total data stored in a block (bytes, not elements)
            /// @author WVN
            size_t GetArrayBytesPerBlock(void);
            
            /// @brief The callback function for new nodes
            /// @param newChildNodes The newly created child nodes to be handled
            /// @param newParentNodes The newly refined parent nodes to be handled
            /// @author WVN
            virtual void OnPostRefinementCallback(std::vector<RefinementTreeNode*>& newChildNodes, std::vector<RefinementTreeNode*> newParentNodes) override final;
            
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
            
            /// @brief Returns the size (in bytes) of the data stored in a single cell
            /// @author WVN
            size_t GetSingleCellSizeBytes(void);
        
            /// @brief Allows for direct indexing using a block iterator
            /// @param it The block iterator to index with
            /// @author WVN
            CartesianMeshArrayPointerPair operator [] (BlockIterator& it);
            
            /// @brief Allows for direct indexing using a node
            /// @param it The node to iterate at
            /// @author WVN
            CartesianMeshArrayPointerPair operator [] (RefinementTreeNode* node);
            
            /// @brief Returns a BlockInfo struct computed from the given block
            /// @param node The block to have info returned for
            /// @author WVN
            BlockInfo GetBlockInfo(RefinementTreeNode* node);
            
            /// @brief \see IBlockIterable::GetRefinementBlockObject
            /// @author WVN
            RefinementBlock* GetRefinementBlockObject(void);
            
            /// @brief Performs data exchanges to and from neighboring blocks
            /// @author WVN
            void Exchange(void);
            
            /// @brief Returns filter
            /// @author WVN
            NodeFilter_t GetFilter(void);
            
            /// @brief Returns the data exchange patterns for this array
            /// @author WVN
            virtual DataExchangePattern* GetExchangePattern(void) override final;
            
            /// @brief Begin() overload for range iteration
            /// @author WVN
            std::vector<RefinementTreeNode*>::iterator begin() noexcept;
            
            /// @brief const Begin() overload for range iteration
            /// @author WVN
            std::vector<RefinementTreeNode*>::const_iterator begin() const noexcept;
            
            /// @brief End() overload for range iteration
            /// @author WVN
            std::vector<RefinementTreeNode*>::iterator end() noexcept;
            
            /// @brief constant End() overload for range iteration
            /// @author WVN
            std::vector<RefinementTreeNode*>::const_iterator end() const noexcept;
            
            /// @author WVN
            /// @brief Returns the base pointer for a node, and NULL if there is not one
            void* GetNodePointerWithNullDefault(RefinementTreeNode* node);
            
            /// @author WVN
            /// @brief Returns the mesh handler
            CartesianMesh* Mesh(void);
            
            /// @brief Reads the data from the parallel file
            /// @param file The file to read from
            /// @author WVN
            virtual void ReadFromFile(ParallelFile& file) override final;
            
            /// @brief Writes the data to the parallel file
            /// @param file The file to write to
            /// @author WVN
            virtual void WriteToFile(ParallelFile& file) override final;
            
            /// @brief Exports the array to an externally-readable format
            /// @param directory The directory to output the file to
            /// @param fileTitle The base name of the file, with no file extension
            /// @author WVN
            void ExportFile(std::string directory, std::string fileTitle);
            
            /// @brief Returns the base type of this array
            /// @author WVN
            CmfArrayType::CmfArrayType GetElementType(void)
            {
                return elementType;
            }
            
            /// @brief Returns the device responsible for a given block
            /// @param node the node to get the device for
            /// @author WVN
            ComputeDevice GetBlockDevice(RefinementTreeNode* node);
            
            /// @brief Interpolates an array from the mesh this array is defined over onto a provided one
            /// @param destination The mesh to interpolate onto
            /// @author WVN
            CartesianMeshArray& InterpolateArray(CartesianMesh& destination);
            
        private:
            
            /// @brief Populates the provided buffer object with the necessary information for parallel IO
            /// @param buf The buffer to populate
            /// @author WVN
            void GetParallelDataBuffer(ParallelDataBuffer& buf);
            
            /// @author WVN
            /// @brief Returns true if the iterable object has this node in its parallel partition
            /// @param node The node to check
            virtual bool ParallelPartitionContainsNode(RefinementTreeNode* node) override final;
            
            /// @brief Writes the nodes over which this variable is defined to the file
            /// @param file The file to write to
            /// @author WVN
            void WriteFilterToFile(ParallelFile& file);
            
            /// @brief Verifies that the filtered nodes are the same between the current variable and the provided file
            /// @param file The file to read from
            /// @author WVN
            void VerifyFilterFromFile(ParallelFile& file);
            
            /// @brief Allocates the underlying pointer
            /// @author WVN
            void AllocateInitialBlocks(void);
            
            /// @brief Populates definedNodes and allocatedNodes. Does not allocate memory
            /// @author WVN
            void GetDefinedNodesAndAllocatedNodes(void);
            
            /// @brief Creates the exchange pattern for this array
            /// @author WVN
            void CreateExchangePattern(void);
            
            /// @brief A filter defining the blocks that this variable is defined over
            NodeFilter_t filter;
            
            /// @brief The handler responsible for this array
            CartesianMeshArrayHandler* handler;
            
            /// @brief The rank of a single element of this array
            int rank;
            
            /// @brief A map that holds the pointers for each block
            std::map<RefinementTreeNode*, void*> pointerMap;
            
            /// @brief Tells whether or not the underlying pointer is allocated or not
            bool isAllocated;
            
            /// @brief The type of a single element
            CmfArrayType::CmfArrayType elementType;
            
            /// @brief The parallel exchange pattern for this mesh array
            DataExchangePattern* exchangePattern;
            
            /// @brief indicates if it is necessary to delete the meshBuffer
            bool deleteMeshBuffer;
            
            /// @brief The mesh buffer object for CPU allocation
            CartesianMeshBuffer* meshBuffer;
            
            /// @brief Nodes over which this variable is defined. It is not 
            /// necessarily true that the variable has an allocated array over each of these.
            std::vector<RefinementTreeNode*> definedNodes;
            
            /// @brief Nodes over which this variable is allocated. This is a subset
            /// of definedNodes.
            std::vector<RefinementTreeNode*> allocatedNodes;
            
            template <typename arType, const int elementRank> friend struct BlockArray;
            friend struct RawUnwrappedBlockArray;
            friend class CartesianMeshExchangeHandler;
    };
}

#endif