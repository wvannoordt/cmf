#ifndef ICMF_DATABASE_READ_WRITE_OBJECT_H
#define ICMF_DATABASE_READ_WRITE_OBJECT_H
#include <string>
#include "ObjectList.h"
namespace cmf
{
    /// @brief Represents an object that can read from and written to a CMF data file
    /// @author WVN
    class ICmfDataBaseReadWriteObject
    {
        public:
            
            /// @brief Constructor
            /// @author WVN
            ICmfDataBaseReadWriteObject(void);
            
            /// @brief Destructor
            /// @author WVN
            virtual ~ICmfDataBaseReadWriteObject(void){};
            
            /// @brief Returns a string that uniquely identifies the current object in a database
            /// @author WVN
            virtual std::string DataBaseName(void)=0;
            
            /// @brief Checks if the given list contains all of the prerequisite objects for the current object to be added
            /// @param listToCheck The list of objects to check
            ///@author WVN
            bool RequiredObjectsAreInList(ObjectList<ICmfDataBaseReadWriteObject*>& listToCheck, std::string& missingObjects);
            
            /// @brief Adds the set of prerequisite objects to objectsRequiredBeforeAddingToDataBase
            ///@author WVN
            virtual void SetRequiredPrereqtuisiteDataBaseObjects(void)=0;
            
            /// @brief Adds the set of automatically added objects to objectsToAutomaticallyAddWhenAddingToDataBase
            ///@author WVN
            virtual void SetAutomaticallyAddedObjects(void)=0;
        
        protected:
            ///@brief A list of database objects that must be found in a database before this object is allowed to be added.
            ///For example, a mesh array may not be added to a database without the mesh first being in the database
            ObjectList<ICmfDataBaseReadWriteObject*> objectsRequiredBeforeAddingToDataBase;
            
            /// @brief a list of objects that are automatically added when this object is added to a database.
            ObjectList<ICmfDataBaseReadWriteObject*> objectsToAutomaticallyAddWhenAddingToDataBase;
            
            friend class CmfDataBase;
    };
}

#endif