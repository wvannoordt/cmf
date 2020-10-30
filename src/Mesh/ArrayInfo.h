#ifndef CMF_ARRAY_INFO_H
#define CMF_ARRAY_INFO_H
#ifndef MAX_RANK
#define MAX_RANK 10
#endif
#include <string>
namespace cmf
{    
    /// @brief Contains (mesh-independent) information about an array
    /// @author WVN
    struct ArrayInfo
    {
        /// @brief The name of the array
        std::string name;
        
        /// @brief The rank of the array
        int rank;
        
        /// @brief The dimensions of the array
        int dimensions[MAX_RANK];
    };
}

#endif