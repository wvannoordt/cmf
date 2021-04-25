#include "CmfDataBase.h"
#include "CmfScreen.h"
#include "StringUtils.h"
#include "CmfPrint.h"
#include "CmfError.h"
#include "ParallelGroup.h"
#include "Path.h"
#include "ParallelFile.h"
#include "Timestamp.h"
#include "Utils.hx"
#include "PTL.h"
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
            this->AugmentHash(itemName);
            return *(databaseItems[idx]);
        }
        else
        {
            objectNames.Add(itemName);
            CmfDataBaseItem* newItem = new CmfDataBaseItem(itemName, NULL);
            objectFilenames.Add(CmfDataBaseItem::NullFileName());
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
        std::string timeStampStr = strformat("//Database {} written on: {}", infoFileName, GetTimeString());
        PTL::PropertyTree infoTree;
        infoTree["DataBaseInfo"]["numObjects"] = databaseItems.Size();
        infoTree["DataBaseInfo"]["bigEndian"]  = (MachineIsBigEndian()?"true":"false");
        for (auto& item:databaseItems)
        {
            size_t idx = databaseItems[item];
            std::string objIdentifier = strformat("Obj{}", ZFill(idx, 7));
            auto& section = infoTree["Objects"][objIdentifier];
            section["directory"] = this->directory;
            section["filename"]  = objectFilenames[idx];
            section["name"]      = item->Name();
        }
        std::stringstream infoStream;
        infoStream << timeStampStr << std::endl;
        infoTree.Serialize(infoStream);
        infoFile.Write(infoStream.str());
        infoFile.Close();
    }
    
    void CmfDataBase::Write(std::string databaseTitle)
    {
        WriteLine(1, strformat("Outputting database: \"{}\"", databaseTitle));
        
        //Loop through the current items and generate the file names for this database instance
        for (auto& item:databaseItems)
        {
            size_t idx = databaseItems[item];
            std::string objFilename = databaseTitle + "." + objectNames[idx] + ".csd";
            objectFilenames[idx] = objFilename;
            Path absolutePath(directory);
            absolutePath += objFilename;
            item->Filename() = absolutePath.Str();
        }
        
        //Create the main database information file
        Path infoFilePath(directory);
        std::string filename = databaseTitle + ".ptl";
        infoFilePath += filename;
        this->WriteDataBaseInfoFile(infoFilePath.Str());
        
        //Loop through the objects again and output them to files
        for (auto& item:databaseItems)
        {
            size_t idx = databaseItems[item];
            WriteLine(3, strformat("Output: \"{}\" to \"{}\"", objectNames[idx], item->Filename()));
            ParallelFile objectFile(this->group);
            if (item->Object() == NULL)
            {
                objectFile.Close();
                CmfError(strformat("Attempted to write object \"{}\" to \"{}\", but found a numm object", objectNames[idx], item->Filename()));
            }
            objectFile.Open(item->Filename());
            item->Object()->WriteToFile(objectFile);
            objectFile.Close();
        }
        group->Synchronize();
    }
}