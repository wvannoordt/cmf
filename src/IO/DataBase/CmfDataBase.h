#ifndef CMF_DATABASE_H
#define CMF_DATABASE_H
#include <string>
#include <vector>
#include "ICmfDataBaseReadWriteObject.h"
#include "ObjectList.h"
namespace cmf
{
    /// @brief A class that represents a complete state of CMF. This can be written to disk or read from disk, and contains
    /// a collection of generic input/output files
    /// @author WVN
    class CmfDataBase
    {
        public:
            /// @brief Constructor
            /// @param directory The directory_in where this data file will output to. It is possible that the object will create subdirectories
            /// @author WVN
            CmfDataBase(std::string directory_in);
            
            /// @brief Constructor
            /// @author WVN
            CmfDataBase(void);
            
            /// @brief Destructor
            /// @author WVN
            ~CmfDataBase(void);
            
            /// @brief Steam operator for adding object to database
            /// @param os The current stream object
            /// @param newObject The object to add to the database
            /// @author WVN
            CmfDataBase& operator<<(ICmfDataBaseReadWriteObject& newObject);
        
        private:
            
            /// @brief The builder function
            /// @param directory The directory_in where this data file will output to. It is possible that the object will create subdirectories
            /// @author WVN
            void Builder(std::string directory_in);
            
            /// @brief The directory where files (and possible subdirectories) will be output
            std::string directory;
            
            /// @brief a list of objects that are contained within the current database
            ObjectList<ICmfDataBaseReadWriteObject*> databaseObjects;
    };
}

#endif