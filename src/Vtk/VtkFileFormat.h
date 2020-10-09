#ifndef VTK_FILEFORMAT_H
#define VTK_FILEFORMAT_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "VtkElement.h"

namespace cmf
{
    namespace VtkFormatType
    {
        enum VtkFormatType
        {
            ascii,
            binary
        };
    }

    /// @brief Class representing a VTK file format type
    /// @author WVN
    class VtkFileFormat : public VtkElement
    {
        public:
            /// @brief Constructor for VtkHeader
            /// @param formType_in Format (binary/ASCII) of file to write
            /// @author WVN
            VtkFileFormat(VtkFormatType::VtkFormatType formType_in);
            
            /// @brief Empty constructor for VtkFileFormat
            /// @author WVN
            VtkFileFormat(void);
            
            /// @brief Destructor for VtkFileFormat
            /// @author WVN
            ~VtkFileFormat(void);
            
            /// @brief Writes file format to file.
            /// @param myfile File stream to be written to 
            /// @author WVN
            void WriteToFile(std::ofstream & myfile);
            
            /// @brief Reads file format from file.
            /// @param myfile File stream to be read from
            /// @author WVN
            void ReadFromFile(std::ofstream & myfile);
        private:
            
            /// @brief File format type
            VtkFormatType::VtkFormatType formType;
    };
}

#endif
