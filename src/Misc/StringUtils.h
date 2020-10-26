#ifndef CMF_STRING_UTIL_H
#define CMF_STRING_UTIL_H

#include <string>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

/// @brief Gets file extension from a file name
/// @param filename The name of the file
/// @author WVN
static inline std::string GetFileExtension(std::string filename)
{
    size_t dot = filename.find_last_of(".");
    if (dot == std::string::npos) return "";
    return filename.substr(dot+1, filename.length()-dot);
}

/// @brief Checks if a file exists
/// @param filename The name of the file
/// @author WVN
static inline bool FileExists(std::string filename)
{
    std::ifstream f(filename.c_str());
    return f.good();
}

static inline std::string NiceCommaString(size_t num)
{
    std::string numstr = std::to_string(num);
    std::string output = "";
    for (int i = numstr.length()-1; i >= 0; i--)
    {
        output = numstr[i] + output;
        if ((i!=0)&&(i!=numstr.length()-1)&&((numstr.length()-i)%3)==0) output = "," + output;
    }
    return output;
}


#endif