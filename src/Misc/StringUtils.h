#ifndef CMF_STRING_UTIL_H
#define CMF_STRING_UTIL_H

#include <string>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <sstream>
#include <iostream>
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

static inline std::string UnitString(size_t quantity, std::string unit, size_t pitch)
{
    int numOrders = 0;
    std::string prefixes[7];
    prefixes[0] = "";
    prefixes[1] = "K";
    prefixes[2] = "M";
    prefixes[3] = "G";
    prefixes[4] = "T";
    prefixes[5] = "P";
    prefixes[6] = "P";
    double dq = (double)quantity;
    while (dq > pitch)
    {
        numOrders++;
        dq/=pitch;
        if (numOrders==6) break;
    }
    return std::to_string(dq) + " " + prefixes[numOrders] + unit;
}

static inline std::string UnitString(size_t quantity, std::string unit)
{
    return UnitString(quantity, unit, 1000);
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

static inline std::string PtrToStr(const void* ptr)
{
    std::ostringstream address;
    address << ptr;
    std::string name = address.str();
    return name;
}


#endif