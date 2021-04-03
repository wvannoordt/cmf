#include "CmfDataBase.h"
#include "CmfScreen.h"
#include "StringUtils.h"
#include "CmfPrint.h"
#include "CmfError.h"
#include "ParallelGroup.h"
#include "Path.h"
#include "ParallelFile.h"
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
        //Get the name of the new object
        std::string newObjectName = newObject->DataBaseName();
        
        //Check to see that the new object is associated with the same parallel group as the database
        if (newObject->HasParallelGroup())
        {
            // Throw error if this object belongs to a different parallel group
            ParallelGroup* objectGroup = newObject->GetDatabaseParallelGroup();
            if (objectGroup != group)
            {
                CmfError(strformat("Cannot add object \"{}\" to database: object is associated with ParallelGroup {}, but database is associated with ParallelGroup {}", newObjectName, objectGroup, group));
            }
        }
        
        //Check to see that all required prerequisite objects are already in the list
        std::string missingObjects;
        if (!newObject->RequiredObjectsAreInList(databaseObjects, missingObjects))
        {
            CmfError(strformat("Cannot add object \"{}\" to database, missing the following prerequisite objects:\n{}", newObjectName, missingObjects));
        }
        
        //Skip if database already has object (in the case the multiple objects require the same object)
        if (!databaseObjects.Has(newObject))
        {
            databaseObjects.Add(newObject);
            
            //Check for name collisions
            if (objectNames.Has(newObjectName))
            {
                CmfError(strformat("Attempted to add database object with duplicate token \"{}\"", newObjectName));
            }
            
            //Add object and update the object hash
            objectNames.Add(newObjectName);
            this->AugmentHash(newObjectName);
            WriteLine(4, strformat("Added object \"{}\" to database", newObjectName));
            
            //Add objects that are automatically added when this object is added
            for (auto& obj: newObject->objectsToAutomaticallyAddWhenAddingToDataBase)
            {
                AddDataBaseObject(obj);
            }
        }
        
        //Crash if the database is out of sync on the associated parallel group
        if (!group->HasSameValue(this->GetHash()))
        {
            CmfError("Database hash check failed: this means that some ranks are writing different objects to the same database");
        }
        group->Synchronize();
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
        if (!group->HasSameValue(this->GetHash()))
        {
            CmfError("Database hash check failed: this is likely because different ranks are attempting to write a database in separate directories");
        }
        group->Synchronize();
    }
    
    
    void CmfDataBase::Write(std::string databaseTitle)
    {
        Path outputPath(directory);
        std::string filename = databaseTitle + ".csd";
        outputPath += filename;
        WriteLine(1, strformat("Outputting database: \"{}\"", outputPath));
        ParallelFile outputFile(this->group);
        outputFile.Open(outputPath.Str());
        
        for (auto& obj:databaseObjects)
        {
            WriteLine(4, strformat("Write object \"{}\" to database \"{}\"", obj->DataBaseName() , outputPath));
            // obj->WriteToFile(outputFile);
        }
        
        outputFile.Close();
        
        group->Synchronize();
    }
}