#ifndef CMF_SVG_ATTR_MOD_H
#define CMF_SVG_ATTR_MOD_H
#include "SvgAttributes.h"
namespace cmf
{
    /// @brief A class that automatically modifies an attribute list
    /// @author WVN
    class SvgAttributeModifier
    {
        public:
            /// @brief Adds attributes from the derived class to the attribute list
            /// @author WVN
            virtual void CreateInheritedAttributes(SvgAttributes& attr) = 0;
    };
}

#endif