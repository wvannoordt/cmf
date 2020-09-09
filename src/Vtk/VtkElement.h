#ifndef VTK_ELEM_H
#define VTK_ELEM_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>

namespace Anaptric
{
    class VtkElement
    {
        public:
            VtkElement(void){elementType="NULL";}
            ~VtkElement(void){}
            virtual void WriteToFile(std::ofstream & myfile){}
            virtual void ReadFromFile(std::ofstream & myfile){}
        protected:
            std::string elementType;
    };
}

#endif