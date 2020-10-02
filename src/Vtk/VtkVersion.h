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
    class VtkVersion : public VtkElement
    {
        public:
            VtkVersion(void);
            VtkVersion(int major_in, int minor_in);
            ~VtkVersion(void);
            void WriteToFile(std::ofstream & myfile);
            void ReadFromFile(std::ofstream & myfile);
        private:
            int major, minor;
    };
}

#endif
