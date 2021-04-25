#ifndef CMF_DATABASE_H
#define CMF_DATABASE_H
#include <string>
#include <vector>
#include "ICmfDataBaseReadWriteObject.h"
#include "ObjectList.h"
#include "ICmfHashable.h"
#include "ParallelGroup.h"
#include "CmfDataBaseItem.h"
namespace cmf
{
    /// @brief A class that represents a complete state of CMF. This can be written to disk or read from disk, and contains
    /// a collection of generic input/output files
    /// @author WVN
    class CmfDataBase : public ICmfHashable
    {
        public:
            /// @brief Constructor
            /// @param directory The directory_in where this data file will output to. It is possible that the object will create subdirectories
            /// @param group_in The parallel group that the database is defined for
            /// @author WVN
            CmfDataBase(std::string directory_in, ParallelGroup* group_in);
            
            /// @brief Constructor
            /// @param group_in The parallel group that the database is defined for
            /// @author WVN
            CmfDataBase(ParallelGroup* group_in);
            
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
            
            /// @brief Returns the CmfDataBaseItem with the provided name, or creates a new one pointing to a NULL ICmfDataBaseReadWriteObject if the
            /// item doesn't exist in the list
            /// @param itemName the name of the desired item
            /// @author WVN
            CmfDataBaseItem& operator [] (std::string itemName);
            
            /// @brief Writes out the database using the given title. The title should not include a file extension
            /// as multiple files may be written. The title should also not include the directory
            /// @author WVN
            void Write(std::string databaseTitle);
        
        private:
            
            /// @brief Writes the database information file
            /// @param infoFileName The name of the database information file
            /// @author WVN
            void WriteDataBaseInfoFile(std::string infoFileName);
            
            /// @brief The builder function
            /// @param directory The directory_in where this data file will output to. It is possible that the object will create subdirectories
            /// @param group_in The parallel group that the database is defined for
            /// @author WVN
            void Builder(std::string directory_in, ParallelGroup* group_in);
            
            /// @brief The directory where files (and possible subdirectories) will be output
            std::string directory;
            
            /// @brief a list of items that are contained within the current database
            ObjectList<CmfDataBaseItem*> databaseItems;
            
            /// @brief A list of the object names
            ObjectList<std::string> objectNames;
            
            /// @brief A list of the object names
            ObjectList<std::string> objectFilenames;
            
            /// @brief The parallel group that the database is defined for (in practice, the parallelGroup that the contained mesh is defined over)
            ParallelGroup* group;
    };
}

#endif