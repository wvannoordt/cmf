#ifndef CMF_GC_H
#define CMF_GC_H
#include <map>
#include "Config.h"
#include "MemSpace.h"
#include <cstddef>

namespace cmf
{
    namespace GlobalStackAllocationParam
    {
        /// @brief Allows (or disallows) stack allocation in this translation unit. It is not recommended to set this directly, but
        /// rather to use the SetStackAllocationAllowed function.
        static bool allowStackAllocationHere = true;
    }
    
    /// @brief Sets GlobalStackAllocationParam::allowStackAllocationHere
    /// @param input The value to set
    /// @author WVN
    static void SetStackAllocationAllowed(bool input) {GlobalStackAllocationParam::allowStackAllocationHere = input;}
    
    /// @brief A class for primitive garbage collection to promote memory safety
    /// @author WVN
    class CmfGC
    {
        public:
            /// @brief Constructor for CmfGC class
            /// @author WVN
            CmfGC(void);
            
            /// @brief Destructor for CmfGC class. Frees all un-freed resources
            /// @author WVN
            ~CmfGC(void);
            
            /// @brief A wrapper for malloc(). Used via the "Cmf_Alloc" macro
            /// @param size Size to allocate
            /// @param file The name of the file where this function is called from
            /// @param line The line number in the file where this function is called from
            /// @param allowStackAllocation Indicates whether or not the memory can be allowed to be allocated on the stack
            /// \pre Note: setting allowStackAllocation to true via SetStackAllocationAllowed will result in a pointer that has been pre-allocated.
            /// This means that valgrind will not catch out-of-bounds errors for that array.
            /// @author WVN
            void* CmfAlloc(size_t size, const char* file, const int line, bool allowStackAllocation);
            
            /// @brief A wrapper for malloc(). Used via the "Cmf_Alloc" macro
            /// @param ptr pointer to free
            /// @param file The name of the file where this function is called from
            /// @param line The line number in the file where this function is called from
            /// @param allowStackAllocation Indicates whether or not the memory can be allowed to be freed from the stack
            /// @author WVN
            void CmfFree(void* ptr, const char* file, const int line, const char* varname, bool allowStackAllocation);
            
            /// @brief A wrapper for cudaMalloc(). Used via the "Cmf_GpuAlloc" macro, returns a device-side pointer
            /// @param size Size to allocate
            /// @param deviceNum The device to allocate on
            /// @param file The name of the file where this function is called from
            /// @param line The line number in the file where this function is called from
            /// @author WVN
            void* CmfGpuAlloc(size_t size, const int deviceNum, const char* file, const int line);
            
            /// @brief A wrapper for cudaFree(). Used via the "Cmf_GpuFree" macro
            /// @param ptr pointer to free
            /// @param file The name of the file where this function is called from
            /// @param line The line number in the file where this function is called from
            /// @author WVN
            void CmfGpuFree(void* ptr, const char* file, const int line, const char* varname);
        
        private:
            
            /// @brief Total size allocated by the GC on the CPU
            size_t allocSize;
            
            /// @brief Total size allocated by the GC on the GPU
            size_t allocSizeGPU;
            
            /// @brief total number of allocations on CPU
            int totalAlloc;
            
            /// @brief total number of frees on CPU
            int totalFree;
            
            /// @brief total number of allocations on GPU
            int totalAllocGPU;
            
            /// @brief total number of frees on GPU
            int totalFreeGPU;
            
            /// @brief Emulates a stack for stack-allocated arrays
            char stackBlob[CMF_STACK_BLOB_SIZE];
            
            /// @brief Size of the emulated stack
            size_t stackBlobSize;
            
            /// @brief Emulated stack pointer (index)
            size_t stackBlobIdx;
            
            /// @brief Holds all pointers that have been stack-allocated
            std::map<void*, size_t> stackAllocatedPointers;
    };
    
    /// @brief Global garbage collector
    extern CmfGC cmfGC;
    
    #define Cmf_Alloc(mysize) cmf::CmfAlloc_Internal(mysize, __FILE__, __LINE__,GlobalStackAllocationParam::allowStackAllocationHere)
    #define Cmf_GpuAlloc(mysize, mydevice) cmf::CmfGpuAlloc_Internal(mysize, mydevice, __FILE__, __LINE__)
    #define Cmf_Free(myptr) cmf::CmfFree_Internal(myptr, __FILE__, __LINE__, #myptr,GlobalStackAllocationParam::allowStackAllocationHere)
    #define Cmf_GpuFree(myptr) cmf::CmfGpuFree_Internal(myptr, __FILE__, __LINE__, #myptr)
    
    /// @brief A wrapper for malloc(): used via the "Cmf_Alloc" macro.
    /// @param size Size to allocate
    /// @author WVN
    static void* CmfAlloc_Internal(size_t size, const char* file, const int line, bool allowStackAllocationHere_in) {return cmfGC.CmfAlloc(size, file, line, allowStackAllocationHere_in);}
    
    /// @brief A wrapper for free(): via the "Cmf_Free" macro.
    /// @param ptr Pointer to free
    /// @author WVN
    static void CmfFree_Internal(void* ptr, const char* file, const int line, const char* varname, bool allowStackAllocationHere_in) {cmfGC.CmfFree(ptr, file, line, varname, allowStackAllocationHere_in);}
    
    /// @brief A wrapper for cudaMalloc(): used via the "Cmf_GpuAlloc" macro.
    /// @param size Size to allocate
    /// @author WVN
    static void* CmfGpuAlloc_Internal(size_t size, const int deviceNum, const char* file, const int line) {return cmfGC.CmfGpuAlloc(size, deviceNum, file, line);}
    
    /// @brief A wrapper for cudaFree(): via the "Cmf_GpuFree" macro.
    /// @param ptr Pointer to free
    /// @author WVN
    static void CmfGpuFree_Internal(void* ptr, const char* file, const int line, const char* varname) {cmfGC.CmfGpuFree(ptr, file, line, varname);}
}

#endif