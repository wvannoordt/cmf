#ifndef CMF_DATA_BASE_ITEM_H
#define CMF_DATA_BASE_ITEM_H
#include <string>
namespace cmf
{
    class CmfDataBase;
    
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
            
        private:
            /// @brief the name of this object with respect to the host database
            std::string name;
            
            /// @brief the database containing this object
            CmfDataBase* hostDataBase;
    };
}

#endif