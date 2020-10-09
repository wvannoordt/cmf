#ifndef VTK_HEADER_H
#define VTK_HEADER_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "VtkElement.h"

namespace cmf
{
    /// @brief Class representing a VTK header
    /// @author WVN
    class VtkHeader : public VtkElement
    {
        public:
            /// @brief Empty constructor for VtkHeader
            /// @author WVN
            VtkHeader(void);
            
            /// @brief Constructor for VtkHeader
            /// @param headerValue_in header string to be written to VTK file
            /// @author WVN
            VtkHeader(std::string headerValue_in);
            
            /// @brief Destructor for VtkHeader
            /// @author WVN
            ~VtkHeader(void);
            
            /// @brief Writes header to file.
            /// @param myfile File stream to be written to 
            /// @author WVN
            void WriteToFile(std::ofstream & myfile);
            
            /// @brief Reads header from file.
            /// @param myfile File stream to be read from
            /// @author WVN
            void ReadFromFile(std::ofstream & myfile);
        private:
            std::string headerValue;
    };
}

#endif
