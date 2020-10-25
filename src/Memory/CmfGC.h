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
    };
    
    /// @brief Global garbage collector
    extern CmfGC cmfGC;
}

#endif