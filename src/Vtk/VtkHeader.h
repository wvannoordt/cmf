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
    class VtkHeader : public VtkElement
    {
        public:
            VtkHeader(void);
            VtkHeader(std::string headerValue_in);
            ~VtkHeader(void);
            void WriteToFile(std::ofstream & myfile);
            void ReadFromFile(std::ofstream & myfile);
        private:
            std::string headerValue;
    };
}

#endif
