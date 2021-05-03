#ifndef CMF_GLOBAL_VARIABLES_H
#define CMF_GLOBAL_VARIABLES_H
#include "PTL.h"
#include "GlobalSettings.h"
namespace cmf
{
    //ALL OF THESE VARIABLES MUST BE IMPLEMENTED IN cmf.cpp
    extern PTL::PropertyTree mainInput;
    extern GlobalSettings globalSettings;
}

#endif