#ifndef VTK_HEADER_H
#define VTK_HEADER_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "I_VtkElement.h"

namespace Anaptric
{
    class VtkHeader : public I_VtkElement
    {
        public:
            VtkHeader(void);
            ~VtkHeader(void);
            void WriteToFile(std::ofstream & myfile);
            void ReadFromFile(std::ofstream & myfile);        
    };
}

#endif