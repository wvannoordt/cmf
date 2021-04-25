#include "CmfDataBase.h"
#include "CmfDataBaseItem.h"
#include "StringUtils.h"
namespace cmf
{
    CmfDataBaseItem::CmfDataBaseItem(std::string name_in, CmfDataBase* hostDataBase_in)
    {
        name = name_in;
        hostDataBase = hostDataBase_in;
    }
}