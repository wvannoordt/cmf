#ifndef CARTESIAN_MESH_BUFFER_H
#define CARTESIAN_MESH_BUFFER_H
#include <vector>
#include "CmfArrayType.h"
#include "ICmfMeshBuffer.h"

namespace cmf
{
    /// @brief Handles memory allocation for a Cartesian mesh array
    /// @author WVN
    class CartesianMeshBuffer : public ICmfMeshBuffer
    {
        public:
            
            /// @brief constructor
            /// @param blockArraySize_in the size of each block in elements (not bytes)
            /// @param arrayType_in the type of each element
            /// @author WVN
            CartesianMeshBuffer(size_t blockArraySize_in, CmfArrayType::CmfArrayType arrayType_in);
            
            /// @brief destructor
            /// @author WVN
            ~CartesianMeshBuffer(void);
            
            /// @brief Expands the current memory pool to get enough free memory for the provided number of blocks
            /// @param numBlocks The number of blocks to reserve
            /// @author WVN
            void ReserveBlocks(int numBlocks);
            
            /// @brief Frees all memory in the buffer
            /// @author WVN
            void Clear(void);
            
            /// @brief Returns a pointer to an available block of memory. This function might allocate new blocks.
            /// @author WVN
            void* Claim(void);
            
            /// @brief Returns the size of a single block's memory in bytes
            /// @author WVN
            size_t BlockSizeBytes(void);
            
        private:
            
            /// @brief The list of base pointers for this buffer
            std::vector<void*> pointers;
            
            /// @brief the size (in elements) of each block
            size_t blockArraySize;
            
            /// @brief The type of this array
            CmfArrayType::CmfArrayType arrayType;
            
            /// @brief TEMPORARY IMPLEMENTATION
            size_t tempCounter;
            
            /// @brief the minimum number of blocks that will be reserved at one time, unless on first reservation
            int blockBatchSize;
    };
}

#endif