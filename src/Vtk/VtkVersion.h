#ifndef VTK_VERSION_H
#define VTK_VERSION_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "I_VtkElement.h"

namespace Anaptric
{
    class VtkVersion : public I_VtkElement
    {
        public:
            VtkVersion(void);
            ~VtkVersion(void);
            void WriteToFile(std::ofstream & myfile);
            void ReadFromFile(std::ofstream & myfile);
    };
}

#endif