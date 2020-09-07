#ifndef VTK_ELEM_H
#define VTK_ELEM_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>

namespace Anaptric
{
    class I_VtkElement
    {
        public:
            I_VtkElement(void){elementType="NULL";}
            ~I_VtkElement(void){}
            virtual void WriteToFile(std::ofstream & myfile){}
            virtual void ReadFromFile(std::ofstream & myfile){}
        protected:
            std::string elementType;
    };
}

#endif