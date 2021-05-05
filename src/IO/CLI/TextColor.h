#ifndef CMF_TEXT_COLOR_H
#define CMF_TEXT_COLOR_H
#include <string>

namespace cmf
{
    ///@brief An enumeration used for colored output on the terminal
    namespace AnsiColor
    {
        enum AnsiColor
        {
            black   = 30,
            red     = 31,
            green   = 32,
            yellow  = 33,
            blue    = 34,
            magenta = 35,
            cyan    = 36,
            white   = 37
        };
    }
    
    ///@brief An enumeration used for styled output on the terminal
    namespace AnsiStyle
    {
        enum AnsiStyle
        {
            revert       = 0,
            bold         = 1,
            underline    = 4,
            inverse      = 7,
            boldOff      = 21,
            underlineOff = 24,
            inverseOff   = 27
        };
    }
    
    ///@brief Returns a string with color and style formatting
    ///@param msg The string to format
    ///@param color The color of the text
    ///@param style The style of the text
    ///@author WVN
    std::string ColorFormatString(std::string msg, AnsiColor::AnsiColor color, AnsiStyle::AnsiStyle style);
    
    ///@brief Returns a string with color and style formatting
    ///@param msg The string to format
    ///@param color The color of the text
    ///@param style The style of the text
    ///@author WVN
    std::string ColorFormatString(std::string msg, AnsiColor::AnsiColor color);
    
    ///@brief Returns "Warning" in bold yellow
    ///@author WVN
    std::string WarningStr(void);
}

#endif