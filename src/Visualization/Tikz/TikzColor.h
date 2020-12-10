#ifndef TIKZCOLOR_H
#define TIKZCOLOR_H

#include <string>
#include "Config.h"

namespace cmf
{
    namespace TikzColor
    {
        /// @brief Provides available standard colors in TikZ.
        /// @author WVN
        enum TikzColor
        {
            red,
            green,
            blue,
            cyan,
            magenta,
            yellow,
            black,
            gray,
            darkgray,
            lightgray,
            brown,
            lime,
            olive,
            orange,
            pink,
            purple,
            teal,
            violet,
            white
        };
    }

    /// @brief Provides string representations of standard colors in TikZ.
    /// @param color an integer-cast TikzColor
    /// @author WVN
    inline static std::string TikzColorStr(int color)
    {
        switch (color)
        {
            case TikzColor::red: return "red";
            case TikzColor::green: return "green";
            case TikzColor::blue: return "blue";
            case TikzColor::cyan: return "cyan";
            case TikzColor::magenta: return "magenta";
            case TikzColor::yellow: return "yellow";
            case TikzColor::black: return "black";
            case TikzColor::gray: return "gray";
            case TikzColor::darkgray: return "darkgray";
            case TikzColor::lightgray: return "lightgray";
            case TikzColor::brown: return "brown";
            case TikzColor::lime: return "lime";
            case TikzColor::olive: return "olive";
            case TikzColor::orange: return "orange";
            case TikzColor::pink: return "pink";
            case TikzColor::purple: return "purple";
            case TikzColor::teal: return "teal";
            case TikzColor::violet: return "violet";
            case TikzColor::white: return "white";
        }
        return PTL_AUTO_ENUM_TERMINATOR;
    }
}
#endif
