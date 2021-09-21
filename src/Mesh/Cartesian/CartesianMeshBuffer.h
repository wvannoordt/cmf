#ifndef CARTESIAN_MESH_BUFFER_H
#define CARTESIAN_MESH_BUFFER_H
#include <vector>
#include <set>
#include <deque>
#include <map>
#include "MemSpace.h"
#include "CmfArrayType.h"
#include "ICmfMeshBuffer.h"

namespace cmf
{
    /// @brief Represents a series of block arrays
    /// @author WVN
    struct CartesianDataChunk
    {
        /// @brief The base pointer for this chunk
        void* base = NULL;
        
        /// @brief The number of blocks this chunk has
        int numBlocks = 0;
        
        /// @brief A list of free blocks
        int numberOfVacantBlocks = 0;
        
        /// @brief indicates if base is allocated on GPU
        bool gpu;
    };
    
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
            
            /// @brief Expands the current (CPU) memory pool to get enough free memory for the provided number of blocks
            /// @param numBlocks The number of blocks to reserve
            /// @author WVN
            void ReserveBlocks(int numBlocks);
            
            /// @brief Expands the current memory pool to get enough free memory for the provided number of blocks
            /// @param numBlocks The number of blocks to reserve
            /// @param space The space to allocate in (MemSpace::Cpu or MemSpace::Gpu)
            /// @param gpuDeviceId the ID of the GPU to allocate on (ignored if CPU allocation)
            /// @author WVN
            void ReserveBlocks(int numBlocks, MemSpace::MemSpace space, int gpuDeviceId);
            
            /// @brief Frees all memory in the buffer
            /// @author WVN
            void Clear(void);
            
            /// @brief Returns a (CPU) pointer to an available block of memory. This function might allocate new blocks.
            /// @author WVN
            void* Claim(void);
            
            /// @brief Returns a pointer to an available block of memory. This function might allocate new blocks.
            /// @param space The memory space to get memory from
            /// @param gpuDeviceId the ID of the GPU device (ignored if on CPU)
            /// @author WVN
            void* Claim(MemSpace::MemSpace space, int gpuDeviceId);
            
            /// @brief Marks the provided block as available. Should be treated as freeing the block array
            /// @author WVN
            void Yield(void* ptr);
            
            /// @brief Returns the size of a single block's memory in bytes
            /// @author WVN
            size_t BlockSizeBytes(void);
            
            /// @brief Identifies the chunks with all blocks free and deallocates them. Does not destroy them.
            /// @author WVN
            void ClearVacantChunks(void);
            
        private:
            
            /// @brief The list of base pointers for this buffer
            std::vector<CartesianDataChunk*> chunks;
            
            /// @brief A map of base pointers to blocks (not chunks) to the chunk they belong to
            std::map<void*, CartesianDataChunk*> pointerToChunks;
            
            /// @brief A list of vacant CPU block arrays and their corresponding chunk objects
            std::deque<std::pair<void*, CartesianDataChunk*>> vacantBlocksCpu;
            
            /// @brief A list of vacant GPU block arrays and their corresponding chunk objects
            std::deque<std::pair<void*, CartesianDataChunk*>> vacantBlocksGpu;
            
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