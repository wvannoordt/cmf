#include "CmfDataFile.h"
#include "CmfScreen.h"
namespace cmf
{
    CmfDataFile::CmfDataFile(void)
    {
        Builder(".");
    }
    
    CmfDataFile::CmfDataFile(std::string directory_in)
    {
        Builder(directory_in);
    }
    
    CmfDataFile::~CmfDataFile(void)
    {
        
    }
    
    void CmfDataFile::Builder(std::string directory_in)
    {
        directory = directory_in;
    }
}