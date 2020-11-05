#ifndef CMF_GC_H
#define CMF_GC_H
#include <map>
namespace cmf
{
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
            void* CmfAlloc(size_t size, const char* file, const int line);
            
            /// @brief A wrapper for free()
            /// @param ptr Pointer to free
            /// @author WVN
            void CmfFree(void* ptr, const char* file, const int line, const char* varname);
        
        private:
            
            /// @brief Total size allocated by the GC
            size_t allocSize;
            
            /// @brief total number of allocations
            size_t totalAlloc;
            
            /// @brief total number of frees
            size_t totalFree;
    };
    
    /// @brief Global garbage collector
    extern CmfGC cmfGC;
    
    #define Cmf_Alloc(mysize) cmf::CmfAlloc_Internal(mysize, __FILE__, __LINE__)
    #define Cmf_Free(myptr) cmf::CmfFree_Internal(myptr, __FILE__, __LINE__, #myptr)
    
    /// @brief A wrapper for malloc()
    /// @param size Size to allocate
    /// @author WVN
    static void* CmfAlloc_Internal(size_t size, const char* file, const int line) {return cmfGC.CmfAlloc(size, file, line);}
    
    /// @brief A wrapper for free()
    /// @param ptr Pointer to free
    /// @author WVN
    static void CmfFree_Internal(void* ptr, const char* file, const int line, const char* varname) {cmfGC.CmfFree(ptr, file, line, varname);}
}

#endif