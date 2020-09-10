#ifndef VTK_DATA_H
#define VTK_DATA_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "VtkElement.h"
#include "VtkAttributeCollection.h"

namespace Anaptric
{
    class VtkData : public VtkElement, public VtkHasAttributableCollection
    {
        public:
            VtkData(void);
            ~VtkData(void);
            void WriteToFile(std::ofstream & myfile);
            void ReadFromFile(std::ofstream & myfile);
        private:

    };
}

#endif
