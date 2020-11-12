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
    /// @brief Class representing a general VTK scalar data chunk
    /// @author WVN
    class VtkScalar : public VtkElement, public VtkHasAttributableCollection
    {
        public:
            /// @brief Constructor for VtkScalar
            /// @author WVN
            VtkScalar(void);
            
            /// @brief Destructor for VtkScalar
            /// @author WVN
            ~VtkScalar(void);
            
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
