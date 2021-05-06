#include "ICmfDataBaseReadWriteObject.h"

namespace cmf
{
    ICmfDataBaseReadWriteObject::ICmfDataBaseReadWriteObject(void)
    {
        
    }
    
    bool ICmfDataBaseReadWriteObject::HasParallelGroup(void)
    {
        return false;
    }
    
    ParallelGroup* ICmfDataBaseReadWriteObject::GetDatabaseParallelGroup(void)
    {
        return NULL;
    }
    
    void ICmfDataBaseReadWriteObject::ReadFromFile(ParallelFile& file)
    {
        // Do nothing
    }
    
    void ICmfDataBaseReadWriteObject::WriteToFile(ParallelFile& file)
    {
        // Do nothing
    }
}