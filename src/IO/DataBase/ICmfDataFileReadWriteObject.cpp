#include "ICmfDataBaseReadWriteObject.h"

namespace cmf
{
    ICmfDataBaseReadWriteObject::ICmfDataBaseReadWriteObject(void)
    {
        
    }
    
    bool ICmfDataBaseReadWriteObject::HasParallelGroup(void)
    {
        return true;
    }
    
    ParallelGroup* ICmfDataBaseReadWriteObject::GetDatabaseParallelGroup(void)
    {
        return NULL;
    }
    
    void ICmfDataBaseReadWriteObject::ReadInfoFromFile(ParallelFile& file)
    {
        // Do nothing
    }
    
    void ICmfDataBaseReadWriteObject::ReadDataFromFile(ParallelFile& file)
    {
        // Do nothing
    }
    
    void ICmfDataBaseReadWriteObject::WriteInfoToFile(ParallelFile& file)
    {
        // Do nothing
    }
    
    void ICmfDataBaseReadWriteObject::WriteDataToFile(ParallelFile& file)
    {
        // Do nothing
    }
    
    
    bool ICmfDataBaseReadWriteObject::RequiredObjectsAreInList(ObjectList<ICmfDataBaseReadWriteObject*>& listToCheck, std::string& missingObjects)
    {
        missingObjects = "";
        bool hasAll = true;
        int notFoundCount = 0;
        for (auto& obj: objectsRequiredBeforeAddingToDataBase)
        {
            bool hasObject = listToCheck.Has(obj);
            hasAll = hasAll && hasObject;
            if (!hasObject)
            {
                if (notFoundCount!= 0) missingObjects += "\n";
                missingObjects += "  >>  ";
                missingObjects += obj->DataBaseName();
                notFoundCount++;
            }
        }
        return hasAll;
    }
}