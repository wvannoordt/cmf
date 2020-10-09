#ifndef VTK_VERSION_H
#define VTK_VERSION_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "VtkElement.h"

namespace cmf
{
    /// @brief A class representing a major and minor VTK version
    /// @author WVN
    class VtkVersion : public VtkElement
    {
        public:
            /// @brief Constructor for the VtkVersion
            /// @author WVN
            VtkVersion(void);
            
            /// @brief /// @brief Constructor for the VtkVersion
            /// @param major_in Major version
            /// @param minor_in Minor version
            /// @author WVN
            VtkVersion(int major_in, int minor_in);
            
            /// @brief Destructor for the VtkVersion
            /// @author WVN
            ~VtkVersion(void);
            
            /// @brief Writes version to file.
            /// @param myfile File stream to be written to 
            /// @author WVN
            void WriteToFile(std::ofstream & myfile);
            
            /// @brief Reads version from file.
            /// @param myfile File stream to be read from
            /// @author WVN
            void ReadFromFile(std::ofstream & myfile);
            
        private:
            /// @brief Major version
            int major;
            
            /// @brief Minor version
            int minor;
    };
}

#endif
