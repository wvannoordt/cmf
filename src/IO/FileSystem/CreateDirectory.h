#ifndef CMF_CREATE_DRIECTORY
#define CMF_CREATE_DRIECTORY
#include "Path.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

namespace cmf
{
    /// @brief Creates a directory with the given path
    /// @param directoryPath the path to create a directory to
    /// @author WVN
    static inline bool CreateDirectory(Path& directoryPath)
    {
        int check = mkdir(directoryPath.Str().c_str(), 0755);
        return check==0;
    }
}

#endif