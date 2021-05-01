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
    
    CmfDataBaseItem& CmfDataBase::GetDataBaseItemAndAddIfNotFound(std::string itemName)
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
    
    CmfDataBaseItem& CmfDataBase::operator [] (std::string itemName)
    {
        return this->GetDataBaseItemAndAddIfNotFound(itemName);
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
    
    std::string CmfDataBase::GetValidObjectsAsString(void)
    {
        std::string output = "";
        int i = 0;
        for (auto obj:databaseItems)
        {
            if (obj->Filename() != CmfDataBaseItem::NullFileName())
            {
                std::string prefix = ((i++==0)?"":"\n");
                output += (prefix+obj->Name());
            }
        }
        return output;
    }
    
    std::string CmfDataBase::GetDataBaseInfoFileName(std::string databaseTitle)
    {
        Path infoFilePath(directory);
        std::string filename = databaseTitle + ".ptl";
        infoFilePath += filename;
        return infoFilePath.Str();
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
    
    void CmfDataBase::ReadDataBaseInfoFile(std::string infoFileName)
    {
        PTL::PropertyTree infoTree;
        infoTree.Read(infoFileName);
        int numDataBaseObjects = infoTree["DataBaseInfo"]["numObjects"];
        bool inputDataBaseIsBigEndian = infoTree["DataBaseInfo"]["bigEndian"];
        if (inputDataBaseIsBigEndian != MachineIsBigEndian())
        {
            CmfError(strformat("Attempted to read a {}-endian database on a {}-endian machine: operation currently not supported.", inputDataBaseIsBigEndian?"big":"little", MachineIsBigEndian()?"big":"little"));
        }
        auto& objects = infoTree["Objects"];
        for (auto& objptr:objects)
        {
            auto& obj = *objptr;
            std::string directoryName = obj["directory"];
            std::string objFilename = obj["filename"];
            std::string itemName = obj["name"];
            auto& item = GetDataBaseItemAndAddIfNotFound(itemName);
            Path filePath(this->directory);
            filePath += objFilename;
            item.Filename() = filePath.Str();
        }
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
        std::string infoFileName = this->GetDataBaseInfoFileName(databaseTitle);
        this->WriteDataBaseInfoFile(infoFileName);
        
        //Loop through the objects again and output them to files
        for (auto& item:databaseItems)
        {
            size_t idx = databaseItems[item];
            WriteLine(3, strformat("Output: \"{}\" to \"{}\"", objectNames[idx], item->Filename()));
            ParallelFile objectFile(this->group);
            if (item->Object() == NULL)
            {
                objectFile.Close();
                CmfError(strformat("Attempted to write object \"{}\" to \"{}\", but found a null object", objectNames[idx], item->Filename()));
            }
            objectFile.Open(item->Filename());
            item->Object()->WriteToFile(objectFile);
            objectFile.Close();
        }
        group->Synchronize();
    }
    
    void CmfDataBase::Read(std::string databaseTitle)
    {
        WriteLine(1, strformat("Reading database: \"{}\"", databaseTitle));
        
        //Read the database information file
        std::string infoFileName = this->GetDataBaseInfoFileName(databaseTitle);
        this->ReadDataBaseInfoFile(infoFileName);
        
        //Loop through the objects and read
        for (auto& item:databaseItems)
        {
            if (item->Filename() == CmfDataBaseItem::NullFileName())
            {
                std::string errorMessage = strformat("Attempted to read object \"{}\" from database \"{}\" in directory \"{}\", but object was not found.", item->Name(), databaseTitle, this->directory);
                errorMessage += " The following objects are in this database:\n" + this->GetValidObjectsAsString();
                CmfError(errorMessage);
            }
            
            if (item->Object() != NULL)
            {
                WriteLine(3, strformat("Reading: \"{}\" from \"{}\"", item->Name(), item->Filename()));
                ParallelFile objectFile(this->group);
                objectFile.Open(item->Filename());
                item->Object()->ReadFromFile(objectFile);
                objectFile.Close();
            }
        }
        
        group->Synchronize();
    }
}