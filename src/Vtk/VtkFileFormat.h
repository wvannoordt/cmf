#ifndef VTK_FILEFORMAT_H
#define VTK_FILEFORMAT_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "I_VtkElement.h"

namespace Anaptric
{
    class VtkFileFormat : public I_VtkElement
    {
        public:
            VtkFileFormat(void);
            ~VtkFileFormat(void);
            void WriteToFile(std::ofstream & myfile);
            void ReadFromFile(std::ofstream & myfile);        
    };
}

#endif