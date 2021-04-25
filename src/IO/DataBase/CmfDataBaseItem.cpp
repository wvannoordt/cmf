#include "CmfDataBase.h"
#include "CmfDataBaseItem.h"
#include "StringUtils.h"
#include "CmfError.h"
namespace cmf
{
    CmfDataBaseItem::CmfDataBaseItem(std::string name_in, CmfDataBase* hostDataBase_in)
    {
        name = name_in;
        hostDataBase = hostDataBase_in;
        dataBaseObject = NULL;
        filename = CmfDataBaseItem::NullFileName();
    }
    
    ICmfDataBaseReadWriteObject* & CmfDataBaseItem::Object(void)
    {
        return dataBaseObject;
    }
    
    void CmfDataBaseItem::operator << (ICmfDataBaseReadWriteObject& obj)
    {
        dataBaseObject = &obj;
    }
    
    void CmfDataBaseItem::operator >> (ICmfDataBaseReadWriteObject& obj)
    {
        dataBaseObject = &obj;
    }
}