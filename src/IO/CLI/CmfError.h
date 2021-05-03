#ifndef CMFERROR_H
#define CMFERROR_H
/// @file CmfError.h
///
/// Provides a consistent way of handling errors and implements CmfException. It is strongly recommended to throw errors this way when developing using CMF. Errors
/// Are thrown via the macro CmfError(myMessage), e.g. CmfError("Not enough data provided");
///
/// @author WVN
 
#include <string>
#include <exception>
#include "CmfOutputStream.h"
#include "TextColor.h"
#ifndef FUNCSUPPORT
#define FUNCSUPPORT 1
#endif
#if(FUNCSUPPORT)
#define CmfError(myMessage) {cmf::CmfError_M(myMessage, __LINE__, __FILE__, __PRETTY_FUNCTION__);}
#else
#define CmfError(myMessage) {cmf::CmfError_M(myMessage, __LINE__, __FILE__);}
#endif
namespace cmf
{
    struct CmfException : public std::exception
    {
        const char* what() const throw()
        {
        	return "(CMF Exception)";
        }
    };
#if(FUNCSUPPORT)
    static inline void CmfError_M(std::string message, const int line, const char* file, const char* func)
#else
    static inline void CmfError_M(std::string message, const int line, const char* file)
#endif
    {
        cmfout << ColorFormatString("------------------------------------------------------", AnsiColor::red, AnsiStyle::bold) << cmfendl;
        cmfout << ColorFormatString(">> CMF Error Thrown <<", AnsiColor::red, AnsiStyle::bold) << "\nFile: " << file << "\nLine: " << line << cmfendl;
#if(FUNCSUPPORT)
        cmfout << "Function: " << func << cmfendl;
#endif
        cmfout << message << cmfendl;
        cmfout << ColorFormatString("------------------------------------------------------", AnsiColor::red, AnsiStyle::bold) << cmfendl;
        throw CmfException();
    }
}
#endif
