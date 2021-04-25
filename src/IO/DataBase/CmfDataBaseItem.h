#ifndef CMF_DATA_BASE_ITEM_H
#define CMF_DATA_BASE_ITEM_H
#include <string>
namespace cmf
{
    class CmfDataBase;
    class ICmfDataBaseReadWriteObject;
    
    /// @brief A class that represents a reference to an ICmfDataBaseReadWriteObject in a CmfDataBase
    /// @author WVN
    class CmfDataBaseItem
    {
        public:
            /// @brief Constructor
            /// @param name_in the name of this object with respect to the host database
            /// @param hostDatabase_in The database that stores this reference
            /// @author WVN
            CmfDataBaseItem(std::string name_in, CmfDataBase* hostDataBase_in);
            
            /// @brief Returns the name of this object
            /// @author WVN
            std::string& Name(void) {return name;}
            
            /// @brief returns filename
            /// @author WVN
            std::string& Filename(void) {return filename;}
            
            /// @brief returns dataBaseObject
            /// @author WVN
            ICmfDataBaseReadWriteObject* & Object(void);
            
            /// @brief Adds an object to the host database at the location referenced by this item
            /// @param obj the object to add to the database
            /// @author WVN
            void operator << (ICmfDataBaseReadWriteObject& obj);
            
            /// @brief Reads an object to the host database from a file
            /// @param obj the object to read from the database
            /// @author WVN
            void operator >> (ICmfDataBaseReadWriteObject& obj);
            
            /// @brief returns a "file name" corresponding to a null file. Default value if no file name is set
            /// @author WVN
            static std::string NullFileName(void) {return "[none]";}
            
        private:
            /// @brief the name of this object with respect to the host database
            std::string name;
            
            /// @brief the database containing this object
            CmfDataBase* hostDataBase;
            
            /// @brief the data base object that this item is associated with
            ICmfDataBaseReadWriteObject* dataBaseObject;
            
            /// @brief the file name containing the data for the object;
            std::string filename;
    };
}

#endif