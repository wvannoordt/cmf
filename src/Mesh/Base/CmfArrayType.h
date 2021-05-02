#ifndef CMF_ARRAY_TYPE_H
#define CMF_ARRAY_TYPE_H
#include "CmfError.h"
namespace cmf
{
    ///@brief An enumeration used to do type-checking for plain-old-data arrays at run-time
    namespace CmfArrayType
    {
        enum CmfArrayType
        {
            CmfInt         =99,
            CmfDouble      =100,
            CmfFloat       =101,
            CmfBool        =102,
            CmfUnsupported =103,
            CmfNullType    =104 //acts as null terminator
        };
    }
    
    /// @brief Returns a string representation of a CmfArrayType
    /// @param atype The type to convert
    /// @author WVN
    static inline std::string CmfArrayTypeToString(CmfArrayType::CmfArrayType atype)
    {
        switch (atype)
        {
            case CmfArrayType::CmfInt:         return "CmfInt";
            case CmfArrayType::CmfDouble:      return "CmfDouble";
            case CmfArrayType::CmfFloat:       return "CmfFloat";
            case CmfArrayType::CmfBool:        return "CmfBool";
            case CmfArrayType::CmfUnsupported: return "CmfUnsupported";
            default:
            {
                CmfError("Attempted to convert CmfArrayType of unknown type to string");
                return 0;
            }
        }
    }
    
    /// @brief Converts a string representation to a CmfArrayType. If the string is invalid, returns CmfNullType
    /// @param str The string to convert
    /// @author WVN
    static inline CmfArrayType::CmfArrayType CmfArrayTypeFromString(std::string str)
    {
        for (int i = (int)CmfArrayType::CmfInt; i <= (int)CmfArrayType::CmfNullType; i++)
        {
            if (str == CmfArrayTypeToString((CmfArrayType::CmfArrayType)i)) return (CmfArrayType::CmfArrayType)i;
        }
        return CmfArrayType::CmfNullType;
    }
    
    
    static inline size_t SizeOfArrayType(CmfArrayType::CmfArrayType aType)
    {
        switch (aType)
        {
            case CmfArrayType::CmfInt:         return sizeof(int);
            case CmfArrayType::CmfDouble:      return sizeof(double);
            case CmfArrayType::CmfFloat:       return sizeof(float);
            case CmfArrayType::CmfBool:        return sizeof(bool);
            case CmfArrayType::CmfUnsupported: return 0;
            default:
            {
                CmfError("Attempted to retrieve size of CmfArrayType of unknown type");
                return 0;
            }
        }
    }
    
}

#endif