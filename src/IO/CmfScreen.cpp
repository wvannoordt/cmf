#include "CmfScreen.h"
#include "CmfOutputStream.h"
namespace cmf
{
    //Probably good to make this a struct...
    bool globalOutputEnabledHere = true;
    bool globalTrackOutputOrigins = false;
    int globalDebugLevel = 0;
    
    void WriteLine_WithFileAndLine(int debugLevel, std::string message, int line, const char* file)
    {
        if (globalOutputEnabledHere && (debugLevel<=globalDebugLevel))
        {
            cmfout << "cmf :: " << message;
            if (globalTrackOutputOrigins) cmfout << "\n >> (debug " << debugLevel << " from file " << file << ", line " << line << ")";
            cmfout << cmfendl;
        }
    }
    
    void WriteLineStd_WithFileAndLine(int debugLevel, std::string message, int line, const char* file)
    {
        if (globalOutputEnabledHere && (debugLevel<=globalDebugLevel))
        {
            std::cout << "cmf :: " << message;
            if (globalTrackOutputOrigins) std::cout << "\n >> (debug " << debugLevel << " from file " << file << ", line " << line << ")";
            std::cout << std::endl;
        }
    }
}