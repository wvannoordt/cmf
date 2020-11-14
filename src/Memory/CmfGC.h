#ifndef CMF_GC_H
#define CMF_GC_H
#include <map>
#include "Config.h"

namespace cmf
{
    namespace GlobalStackAllocationParam
    {
        /// @brief Allows (or disallows) stack allocation in this translation unit
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
            
            /// @brief A wrapper for malloc()
            /// @param size Size to allocate
            /// @author WVN
            void* CmfAlloc(size_t size, const char* file, const int line, bool allowStackAllocation);
            
            /// @brief A wrapper for free()
            /// @param ptr Pointer to free
            /// @author WVN
            void CmfFree(void* ptr, const char* file, const int line, const char* varname, bool allowStackAllocation);
        
        private:
            
            /// @brief Total size allocated by the GC
            size_t allocSize;
            
            /// @brief total number of allocations
            int totalAlloc;
            
            /// @brief total number of frees
            int totalFree;
            
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
    #define Cmf_Free(myptr) cmf::CmfFree_Internal(myptr, __FILE__, __LINE__, #myptr,GlobalStackAllocationParam::allowStackAllocationHere)
    
    /// @brief A wrapper for malloc()
    /// @param size Size to allocate
    /// @author WVN
    static void* CmfAlloc_Internal(size_t size, const char* file, const int line, bool allowStackAllocationHere_in) {return cmfGC.CmfAlloc(size, file, line, allowStackAllocationHere_in);}
    
    /// @brief A wrapper for free()
    /// @param ptr Pointer to free
    /// @author WVN
    static void CmfFree_Internal(void* ptr, const char* file, const int line, const char* varname, bool allowStackAllocationHere_in) {cmfGC.CmfFree(ptr, file, line, varname, allowStackAllocationHere_in);}
}

#endif