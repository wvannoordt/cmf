#ifndef CMF_CREATE_DRIECTORY
#define CMF_CREATE_DRIECTORY
#include "Path.h"
#include "CmfError.h"
#include "TextColor.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <cstring>

//todo: do a better job of this
namespace cmf
{    
    /// @brief Creates a directory with the given path, does not crash if directory exists
    /// @param directoryPath the path to create a directory to
    /// @author WVN
    static inline bool CreateDirectory(Path& directoryPath)
    {
        int check = mkdir(directoryPath.Str().c_str(), 0755);
        if (check != 0)
        {
            auto er = errno;
            if (er==EEXIST) return true;
            std::stringstream ss;
            ss << std::strerror(errno);
            CmfError(strformat(ColorFormatString("System error: {}", AnsiColor::magenta, AnsiStyle::bold), ss.str()));
            return false;
        }
        return true;
    }
    
    /// @brief Creates a directory with the given path, does not crash if directory exists
    /// @param directoryPath the path to create a directory to
    /// @author WVN
    static inline bool CreateDirectory(std::string directoryPath)
    {
        Path dp(directoryPath);
        return CreateDirectory(dp);
    }
}

#endif