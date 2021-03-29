#include "CmfDataBase.h"
#include "CmfScreen.h"
#include "StringUtils.h"
#include "CmfPrint.h"
#include "CmfError.h"
#include "ParallelGroup.h"
namespace cmf
{
    CmfDataBase::CmfDataBase(void)
    {
        Builder(".", &globalGroup);
    }
    
    CmfDataBase::CmfDataBase(std::string directory_in)
    {
        Builder(directory_in, &globalGroup);
    }
    
    CmfDataBase::CmfDataBase(std::string directory_in, ParallelGroup* group_in)
    {
        Builder(directory_in, group_in);
    }
    
    CmfDataBase::CmfDataBase(ParallelGroup* group_in)
    {
        Builder(".", group_in);
    }
    
    void CmfDataBase::AddDataBaseObject(ICmfDataBaseReadWriteObject* newObject)
    {
        std::string newObjectName = newObject->DataBaseName();
        std::string missingObjects;
        if (!newObject->RequiredObjectsAreInList(databaseObjects, missingObjects))
        {
            CmfError(strformat("Cannot add object \"{}\" to database, missing the following prerequisite objects:\n{}", newObjectName, missingObjects));
        }
        if (!databaseObjects.Has(newObject))
        {
            databaseObjects.Add(newObject);
            if (objectNames.Has(newObjectName))
            {
                CmfError(strformat("Attempted to add database object with duplicate token \"{}\"", newObjectName));
            }
            objectNames.Add(newObjectName);
            this->AugmentHash(newObjectName);
            WriteLine(4, strformat("Added object \"{}\" to database", newObjectName));
            for (auto& obj: newObject->objectsToAutomaticallyAddWhenAddingToDataBase)
            {
                AddDataBaseObject(obj);
            }
        }
    }
    
    CmfDataBase& CmfDataBase::operator << (ICmfDataBaseReadWriteObject& newObject)
    {
        this->AddDataBaseObject(&newObject);
        return *this;
    }
    
    CmfDataBase::~CmfDataBase(void)
    {
        
    }
    
    void CmfDataBase::Builder(std::string directory_in, ParallelGroup* group_in)
    {
        group = group_in;
        AugmentHash(directory_in);
        directory = directory_in;
        WriteLine(3, strformat("Defining database in directory \"{}\"", directory));
    }
}