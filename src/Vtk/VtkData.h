#ifndef VTK_DATA_H
#define VTK_DATA_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "I_VtkElement.h"

namespace Anaptric
{
    class VtkData : public I_VtkElement
    {
        public:
            VtkData(void);
            ~VtkData(void);
            void WriteToFile(std::ofstream & myfile);
            void ReadFromFile(std::ofstream & myfile);        
    };
}

#endif