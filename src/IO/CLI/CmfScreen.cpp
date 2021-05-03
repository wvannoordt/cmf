#include "CmfScreen.h"
#include "CmfOutputStream.h"
#include "cmf.h"
#include "TextColor.h"
namespace cmf
{
    const AnsiColor::AnsiColor msgColor = AnsiColor::cyan;
    const AnsiStyle::AnsiStyle msgStyle = AnsiStyle::bold;
    void WriteLine_WithFileAndLine(int debugLevel, std::string message, int line, const char* file)
    {
        if (globalSettings.globalOutputEnabledHere && (debugLevel<=globalSettings.debugLevel))
        {
            cmfout << ColorFormatString("cmf :: ", msgColor, msgStyle) << message;
            if (globalSettings.trackOutputOrigins) cmfout << "\n >> (debug " << debugLevel << " from file " << file << ", line " << line << ")";
            cmfout << cmfendl;
        }
    }
    
    void WriteLineStd_WithFileAndLine(int debugLevel, std::string message, int line, const char* file)
    {
        if (globalSettings.globalOutputEnabledHere && (debugLevel<=globalSettings.debugLevel))
        {
            std::cout << ColorFormatString("cmf :: ", msgColor, msgStyle) << message;
            if (globalSettings.trackOutputOrigins) std::cout << "\n >> (debug " << debugLevel << " from file " << file << ", line " << line << ")";
            std::cout << std::endl;
        }
    }
    
    void ParWriteLine_WithFileAndLine(int debugLevel, std::string message, int line, const char* file)
    {
        if ((debugLevel<=globalSettings.debugLevel))
        {
            cmfout << ColorFormatString("cmf :: ", msgColor, msgStyle) << message;
            if (globalSettings.trackOutputOrigins) cmfout << "\n >> (debug " << debugLevel << " from file " << file << ", line " << line << ")";
            cmfout << cmfendl;
        }
    }
    
    void ParWriteLineStd_WithFileAndLine(int debugLevel, std::string message, int line, const char* file)
    {
        if ((debugLevel<=globalSettings.debugLevel))
        {
            std::cout << ColorFormatString("cmf :: ", msgColor, msgStyle) << message;
            if (globalSettings.trackOutputOrigins) std::cout << "\n >> (debug " << debugLevel << " from file " << file << ", line " << line << ")";
            std::cout << std::endl;
        }
    }
}