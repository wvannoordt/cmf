#ifndef VTK_FILEFORMAT_H
#define VTK_FILEFORMAT_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "VtkElement.h"

namespace Anaptric
{
    namespace VtkFormatType
    {
        enum VtkFormatType
        {
            ascii,
            binary
        };
    }
    
    class VtkFileFormat : public VtkElement
    {
        public:
            VtkFileFormat(VtkFormatType::VtkFormatType formType_in);
            VtkFileFormat(void);
            ~VtkFileFormat(void);
            void WriteToFile(std::ofstream & myfile);
            void ReadFromFile(std::ofstream & myfile);        
        private:
            VtkFormatType::VtkFormatType formType;
    };
}

#endif