#ifndef CMF_CART_PARTITION_TYPE_H
#define CMF_CART_PARTITION_TYPE_H
#include <string>
#include "PTL.h"
namespace cmf
{
    namespace CartesianPartitionType
    {
        enum CartesianPartitionType
        {
            uniform,
            singleGPU,
            unsupported
        };
    }

    inline static std::string CartesianPartitionTypeStr(int refType)
    {
        switch (refType)
        {
            case CartesianPartitionType::uniform:     return "uniform";
            case CartesianPartitionType::singleGPU:   return "singleGPU";
            case CartesianPartitionType::unsupported: return "unsupported";
        }
        return PTL_AUTO_ENUM_TERMINATOR;
    }
}
#endif
