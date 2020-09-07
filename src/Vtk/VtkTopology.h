#ifndef VTK_TOPOLOGY_H
#define VTK_TOPOLOGY_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "I_VtkElement.h"

namespace Anaptric
{
    class VtkTopology : public I_VtkElement
    {
        public:
            VtkTopology(void);
            ~VtkTopology(void);
            void WriteToFile(std::ofstream & myfile);
            void ReadFromFile(std::ofstream & myfile);      
    };
}

#endif