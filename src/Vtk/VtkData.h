#ifndef VTK_DATA_H
#define VTK_DATA_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "VtkElement.h"
#include "VtkAttributeCollection.h"

namespace cmf
{
    /// @brief Class representing a general VTK data chunk
    /// @author WVN
    class VtkData : public VtkElement, public VtkHasAttributableCollection
    {
        public:
            /// @brief Constructor for VtkData
            /// @author WVN
            VtkData(void);
            
            /// @brief Destructor for VtkData
            /// @author WVN
            ~VtkData(void);
            
            /// @brief Write data to file stream
            /// @param myfile Stream to be written to
            /// @author WVN
            void WriteToFile(std::ofstream & myfile);
            
            /// @brief Read data from file stream
            /// @param myfile Stream to be read from
            /// @author WVN
            void ReadFromFile(std::ofstream & myfile);
        private:

    };
}

#endif
