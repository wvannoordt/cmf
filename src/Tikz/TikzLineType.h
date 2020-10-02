#ifndef TIKZLINE_H
#define TIKZLINE_H

#include "Config.h"

namespace cmf
{
    namespace TikzLineType
    {
        enum TikzLineType
        {
            solid,
            dotted,
            denselyDotted,
            looselyDotted,
            dashed,
            denselyDashed,
            looselyDashed,
            dashDotted,
            denselyDashDotted,
            looselyDashDotted,
            dashDotDotted,
            denselyDashDotDotted,
            looselyDashDotDotted
        };

        inline static std::string TikzTypeStr(int lineType)
        {
            switch (lineType)
            {
                case TikzLineType::solid: return "solid";
                case TikzLineType::dotted: return "dotted";
                case TikzLineType::denselyDotted: return "densely dotted";
                case TikzLineType::looselyDotted: return "loosely dotted";
                case TikzLineType::dashed: return "dashed";
                case TikzLineType::denselyDashed: return "densely dashed";
                case TikzLineType::looselyDashed: return "loosely dashed";
                case TikzLineType::dashDotted: return "dashdotted";
                case TikzLineType::denselyDashDotted: return "densely dashdotted";
                case TikzLineType::looselyDashDotted: return "loosely dashdotted";
                case TikzLineType::dashDotDotted: return "dashdotdotted";
                case TikzLineType::denselyDashDotDotted: return "densely dashdotdotted";
                case TikzLineType::looselyDashDotDotted: return "loosely dashdotdotted";
            }
            return PTL_AUTO_ENUM_TERMINATOR;
        }
    }
}
#endif
