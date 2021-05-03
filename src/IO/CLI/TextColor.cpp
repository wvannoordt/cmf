#include "TextColor.h"
#include "CmfGlobalVariables.h"
namespace cmf
{
    std::string ColorFormatString(std::string msg, AnsiColor::AnsiColor color, AnsiStyle::AnsiStyle style)
    {
        std::string output = "";
        if (globalSettings.colorOutput) output += "\033[" + std::to_string((int)style) + ";" + std::to_string((int)color) + "m";
        output += msg;
        if (globalSettings.colorOutput) output += "\033[0m";
        return output;
    }
    
    std::string ColorFormatString(std::string msg, AnsiColor::AnsiColor color)
    {
        return ColorFormatString(msg, color, AnsiStyle::revert);
    }
}