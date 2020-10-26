#include "CmfGC.h"
#include "CmfScreen.h"
#include "StringUtils.h"
#include <iostream>
#include "DebugTools.hx"
namespace cmf
{
    CmfGC cmfGC;
    CmfGC::CmfGC(void){allocSize = 0;totalAlloc=0;totalFree=0;}
    
    CmfGC::~CmfGC(void)
    {
        WriteLineStd(4, "Freeing resources");
        WriteLineStd(4, "Total allocs: " + NiceCommaString(totalAlloc));
        WriteLineStd(4, "Total frees: " + NiceCommaString(totalFree));
    }
    
    void* CmfGC::CmfAlloc(size_t size, const char* file, const int line)
    {
        totalAlloc++;
        allocSize += size;
        WriteLine(5, "Allocating buffer, cumulative size: " + NiceCommaString(allocSize));
        std::string filestr(file);
        WriteLine(6, "From file " + filestr + ", line " + std::to_string(line));
        void* output = malloc(size);
        return output;
    }
    
    void CmfGC::CmfFree(void* ptr, const char* file, const int line, const char* varname)
    {
        totalFree++;
        std::string filestr(file);
        std::string varstr(varname);
        WriteLine(5, "Freeing variable " + varstr);
        WriteLine(6, "From file " + filestr + ", line " + std::to_string(line));
        free(ptr);
    }
}