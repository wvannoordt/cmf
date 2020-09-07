#ifndef VTK_FILE_H
#define VTK_FILE_H

#include <string>
#include <vector>

namespace Anaptric
{
    class VtkFile
    {
        public:
            VtkFile(std::string filename_in);
            ~VtkFile(void);
        private:
            std::string filename;
    };
}

#endif