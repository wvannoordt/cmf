#ifndef CMF_PARALLEL_DATA_BUFFER_H
#define CMF_PARALLEL_DATA_BUFFER_H

namespace cmf
{
    ///@brief Represents a series of pointers (assumed non-aliased), sizes, and
    ///absolute offsets that describe a data buffer in a parallel context
    ///@author WVN
    class ParallelDataBuffer
    {
        public:
            ///@brief Constructor
            ///@author WVN
            ParallelDataBuffer(void) {}
            
            ///@brief Constructor
            ///@author WVN
            ~ParallelDataBuffer(void) {}
            
            ///@brief Adds a pointer, size, and global offset to this dataset
            ///@param ptr the base pointer for the buffer
            ///@param dataSize the size of the data
            ///@param globalOffset the offset (relative to the global position) of this buffer
            ///@author WVN
            void Add(void* ptr, size_t dataSize, size_t globalOffset);
            
        private:
            
            ///@brief list of pointers on this rank
            std::vector<void*> pointers;
            
            ///@brief list of sizes of each buffer located at the corresponding pointer
            std::vector<size_t> sizes;
            
            ///@brief global offsets for the corresponding pointers
            std::vector<size_t> offsets;
            
            friend class ParallelFile;
    };
}

#endif