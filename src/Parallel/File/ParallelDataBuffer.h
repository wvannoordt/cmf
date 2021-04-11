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
            ///@param numElements the size of the data (not in bytes)
            ///@param globalOffset (in elements, not bytes) the offset (relative to the global position) of this buffer
            ///@author WVN
            template <typename bufType> void Add(bufType* ptr, size_t numElements, size_t globalOffset)
            {
                pointers.push_back((void*)ptr);
                sizes.push_back(numElements*sizeof(bufType));
                offsets.push_back(globalOffset*sizeof(bufType));
            }
            
        private:
            
            ///@brief list of pointers on this rank
            std::vector<void*> pointers;
            
            ///@brief list of sizes (in bytes, not elements) of each buffer located at the corresponding pointer
            std::vector<size_t> sizes;
            
            ///@brief global offsets (in bytes, not elements) for the corresponding pointers
            std::vector<size_t> offsets;
            
            friend class ParallelFile;
    };
}

#endif