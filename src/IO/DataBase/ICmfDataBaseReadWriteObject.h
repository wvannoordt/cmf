#ifndef ICMF_DATABASE_READ_WRITE_OBJECT_H
#define ICMF_DATABASE_READ_WRITE_OBJECT_H
#include <string>
#include "ObjectList.h"
#include "ParallelGroup.h"
#include "ParallelFile.h"
namespace cmf
{
    /// @brief Represents an object that can read from and written to a CMF data file.
    /// It is expected that objects that implement this interface are composed of "information",
    /// which is lightweight data such as sizes, strings, names, etc. and "data", which
    /// consists of large arrays of numerics etc
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
            
            /// @brief Indicates whether the object has a ParallelGroup associated with it. Default implementation returns false.
            /// If this returns false, it is assumed that the object is treated in parallel
            ///@author WVN
            virtual bool HasParallelGroup(void);
            
            /// @brief Returns the parallel group for the object, or NULL if it is a serial object
            /// @author WVN
            virtual ParallelGroup* GetDatabaseParallelGroup(void);
            
            /// @brief Reads the object from the parallel file
            /// @param file The file to read from
            /// @author WVN
            virtual void ReadFromFile(ParallelFile& file);
            
            /// @brief Writes the object to the parallel file
            /// @param file The file to write to
            /// @author WVN
            virtual void WriteToFile(ParallelFile& file);
        
        protected:
            
            friend class CmfDataBase;
    };
}

#endif