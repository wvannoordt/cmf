#include "CmfScreen.h"
#include "CmfOutputStream.h"
#include "cmf.h"
namespace cmf
{
    //Probably good to make this a struct...
    bool globalOutputEnabledHere = true;
    bool globalTrackOutputOrigins = false;
    int globalDebugLevel = 0;
    
    void WriteLine_WithFileAndLine(int debugLevel, std::string message, int line, const char* file)
    {
        if (globalSettings.globalOutputEnabledHere && (debugLevel<=globalSettings.debugLevel))
        {
            cmfout << "cmf :: " << message;
            if (globalSettings.trackOutputOrigins) cmfout << "\n >> (debug " << globalSettings.debugLevel << " from file " << file << ", line " << line << ")";
            cmfout << cmfendl;
        }
    }
    
    void WriteLineStd_WithFileAndLine(int debugLevel, std::string message, int line, const char* file)
    {
        if (globalSettings.globalOutputEnabledHere && (debugLevel<=globalSettings.debugLevel))
        {
            std::cout << "cmf :: " << message;
            if (globalSettings.trackOutputOrigins) std::cout << "\n >> (debug " << globalSettings.debugLevel << " from file " << file << ", line " << line << ")";
            std::cout << std::endl;
        }
    }
}