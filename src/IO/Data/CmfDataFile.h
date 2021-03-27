#ifndef CMF_DATA_FILE_H
#define CMF_DATA_FILE_H
#include <string>
#include <vector>
#include "ICmfDataFileReadWriteObject.h"
namespace cmf
{
    /// @brief A class that represents a complete state of CMF. This can be written to disk or read from disk, and contains
    /// a collection of generic input/output files
    /// @author WVN
    class CmfDataFile
    {
        public:
            /// @brief Constructor
            /// @param directory The directory_in where this data file will output to. It is possible that the object will create subdirectories
            /// @author WVN
            CmfDataFile(std::string directory_in);
            
            /// @brief Constructor
            /// @author WVN
            CmfDataFile(void);
            
            /// @brief Destructor
            /// @author WVN
            ~CmfDataFile(void);
        
        private:
            
            /// @brief The builder function
            /// @param directory The directory_in where this data file will output to. It is possible that the object will create subdirectories
            /// @author WVN
            void Builder(std::string directory_in);
            
            /// @brief The directory where files (and possible subdirectories) will be output
            std::string directory;
    };
}

#endif