#include "CmfDataBase.h"
#include "CmfScreen.h"
#include "StringUtils.h"
#include "CmfPrint.h"
#include "CmfError.h"
#include "ParallelGroup.h"
#include "Path.h"
#include "ParallelFile.h"
#include "Timestamp.h"

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
    
    CmfDataBase::~CmfDataBase(void)
    {
        for (auto& item:databaseItems)
        {
            delete item;
        }
    }
    
    CmfDataBaseItem& CmfDataBase::operator [] (std::string itemName)
    {
        if (objectNames.Has(itemName))
        {
            size_t idx = objectNames[itemName];
            return *(databaseItems[idx]);
        }
        else
        {
            objectNames.Add(itemName);
            CmfDataBaseItem* newItem = new CmfDataBaseItem(itemName, NULL);
            databaseItems.Add(newItem);
            return *newItem;
        }
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
        std::string timeStampStr = strformat("Database {} written on: {}", infoFileName, GetTimeString());
        infoFile.Write(timeStampStr);
        infoFile.Write("[database]");
        for (auto& item:databaseItems)
        {
            infoFile.Write("[object]");
            infoFile.Write("name:", item->Name());
            infoFile.Write("[/object]");
        }
        infoFile.Write("[database]");
        infoFile.Close();
    }
    
    void CmfDataBase::Write(std::string databaseTitle)
    {
        Path infoFilePath(directory);
        std::string filename = databaseTitle + ".csd";
        infoFilePath += filename;
        WriteLine(1, strformat("Outputting database: \"{}\"", infoFilePath));
        
        this->WriteDataBaseInfoFile(infoFilePath.Str());
        
        for (auto& item:databaseItems)
        {
            Path infoFilePath(directory);
            std::string objFilename = databaseTitle + ".csd";
            objFilename += filename;
            
            // obj->WriteToFile(outputFile);
        }
        group->Synchronize();
    }
}