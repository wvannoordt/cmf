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
    
    void CmfDataBase::AddDataBaseObject(ICmfDataBaseReadWriteObject* newObject, std::string newObjectName)
    {
        
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
        this->AddDataBaseObject(&newObject, "BROKEN FOR NOW");
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
    
    void CmfDataBase::WriteDataBaseInfoFile(std::string infoFileName)
    {
        ParallelFile infoFile(this->group);
        infoFile.Open(infoFileName);
        for (auto& obj:databaseObjects)
        {
            size_t idx = databaseObjects[obj];
            std::string obname = objectNames[idx];
            infoFile.Write(obname);
        }
        infoFile.Close();
    }
    
    void CmfDataBase::Write(std::string databaseTitle)
    {
        Path infoFilePath(directory);
        std::string filename = databaseTitle + ".csd";
        infoFilePath += filename;
        WriteLine(1, strformat("Outputting database: \"{}\"", infoFilePath));
        
        this->WriteDataBaseInfoFile(infoFilePath.Str());
        
        for (auto& obj:databaseObjects)
        {
            Path infoFilePath(directory);
            std::string objFilename = databaseTitle + ".csd";
            objFilename += filename;
            
            // obj->WriteToFile(outputFile);
        }
        
        group->Synchronize();
    }
}