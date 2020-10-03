#ifndef CMFERROR_H
#define CMFERROR_H

#include <string>
#include <exception>
#include "CmfOutputStream.h"
#define CmfError(myMessage) {cmf::CmfError_M(myMessage, __LINE__, __FILE__);}
namespace cmf
{
    struct CmfException : public std::exception
    {
        const char* what() const throw()
        {
        	return "(CMF Exception)";
        }
    };
    static inline void CmfError_M(std::string message, const int line, const char* file)
    {
        cmfout << "------------------------------------------------------" << cmfendl;
        cmfout << "CMF Error Thrown.\nFile: " << file << "\nLine: " << line << cmfendl;
        cmfout << message << cmfendl;
        cmfout << "------------------------------------------------------" << cmfendl;
        throw CmfException();
    }
}

#endif
