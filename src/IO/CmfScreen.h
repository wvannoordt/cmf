#ifndef CMF_SCREEN_H
#define CMF_SCREEN_H
#include <string>
#define WriteLine(mylevel, mymessage) {cmf::WriteLine_WithFileAndLine((mylevel), (mymessage), __LINE__, __FILE__);}
#define WriteLineStd(mylevel, mymessage) {cmf::WriteLineStd_WithFileAndLine((mylevel), (mymessage), __LINE__, __FILE__);}
namespace cmf
{
    /// @brief Allows screen output if set to true
    extern bool globalOutputEnabledHere;
    
    /// @brief The level of output from CMF
    extern int globalDebugLevel;
    
    /// @brief Displays the file and line number that any screen output is being generated from
    extern bool globalTrackOutputOrigins;
    
    /// @brief Write a message to the screen
    /// @param debugLevel The level of debug output. if the global debug level is set lower than this, the message will not be ouptut
    /// @param message The message to write
    /// @param line The line number where this function is called from
    /// @param file The file where this function is called from
    /// @author WVN
    void WriteLine_WithFileAndLine(int debugLevel, std::string message, int line, const char* file);
    
    /// @brief Write a message to the screen using only standard output stream
    /// @param debugLevel The level of debug output. if the global debug level is set lower than this, the message will not be ouptut
    /// @param message The message to write
    /// @param line The line number where this function is called from
    /// @param file The file where this function is called from
    /// @author WVN
    void WriteLineStd_WithFileAndLine(int debugLevel, std::string message, int line, const char* file);
}

#endif