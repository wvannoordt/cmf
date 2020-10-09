#ifndef VTK_ELEM_H
#define VTK_ELEM_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>

namespace cmf
{
    /// @brief Class representing a general VTK element, essentially an abstract class
    /// @author WVN
    class VtkElement
    {
        public:
            /// @brief Abstract constructor
            /// @author WVN
            VtkElement(void){elementType="NULL";}
            
            /// @brief Abstract destructor
            /// @author WVN
            ~VtkElement(void){}
            
            /// @brief Write content to file stream
            /// @param myfile Stream to be written to
            /// @author WVN
            virtual void WriteToFile(std::ofstream & myfile){}
            
            /// @brief Read content from file stream
            /// @param myfile Stream to be read from
            /// @author WVN
            virtual void ReadFromFile(std::ofstream & myfile){}
        protected:
            
            /// @brief A string representation of the element type
            std::string elementType;
    };
}

#endif
