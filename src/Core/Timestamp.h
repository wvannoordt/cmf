#ifndef CMF_TIMESTAMP_H
#define CMF_TIMESTAMP_H

#include <string>
#include <chrono>
#include <ctime>
#include <algorithm>

namespace cmf
{
    std::string GetTimeString(void)
    {
        auto end = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        auto timeStamp = std::ctime(&end_time);
        std::stringstream str;
        str << timeStamp;
        std::string timeStampStr = str.str();
        timeStampStr.erase(std::remove(timeStampStr.begin(), timeStampStr.end(), '\n'), timeStampStr.end());
        return timeStampStr;
    }
}

#endif