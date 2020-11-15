#ifndef CMF_SCREEN_H
#define CMF_SCREEN_H
#include <string>
#define WriteLine(mylevel, mymessage) {cmf::WriteLine_WithFileAndLine((mylevel), (mymessage), __LINE__, __FILE__);}
#define WriteLineStd(mylevel, mymessage) {cmf::WriteLineStd_WithFileAndLine((mylevel), (mymessage), __LINE__, __FILE__);}
#define ParWriteLine(mylevel, mymessage) {cmf::WriteLine_WithFileAndLine((mylevel), (mymessage), __LINE__, __FILE__);}
#define ParWriteLineStd(mylevel, mymessage) {cmf::WriteLineStd_WithFileAndLine((mylevel), (mymessage), __LINE__, __FILE__);}
namespace cmf
{    
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
    
    /// @brief Write a message to the screen, parallel version. Writes to screen from non-root processes are allowed
    /// @param debugLevel The level of debug output. if the global debug level is set lower than this, the message will not be ouptut
    /// @param message The message to write
    /// @param line The line number where this function is called from
    /// @param file The file where this function is called from
    /// @author WVN
    void ParWriteLine_WithFileAndLine(int debugLevel, std::string message, int line, const char* file);
    
    /// @brief Write a message to the screen using only standard output stream, parallel version. Writes to screen from non-root processes are allowed
    /// @param debugLevel The level of debug output. if the global debug level is set lower than this, the message will not be ouptut
    /// @param message The message to write
    /// @param line The line number where this function is called from
    /// @param file The file where this function is called from
    /// @author WVN
    void ParWriteLineStd_WithFileAndLine(int debugLevel, std::string message, int line, const char* file);
}

#endif