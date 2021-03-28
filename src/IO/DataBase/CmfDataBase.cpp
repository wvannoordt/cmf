#include "CmfDataBase.h"
#include "CmfScreen.h"
#include "StringUtils.h"
namespace cmf
{
    CmfDataBase::CmfDataBase(void)
    {
        Builder(".");
    }
    
    CmfDataBase::CmfDataBase(std::string directory_in)
    {
        Builder(directory_in);
    }
    
    CmfDataBase& CmfDataBase::operator<<(ICmfDataBaseReadWriteObject& newObject)
    {
        databaseObjects.Add(&newObject);
        return *this;
    }
    
    CmfDataBase::~CmfDataBase(void)
    {
        
    }
    
    void CmfDataBase::Builder(std::string directory_in)
    {
        directory = directory_in;
        WriteLine(3, strformat("Defining database in directory \"{}\"", directory));
    }
}