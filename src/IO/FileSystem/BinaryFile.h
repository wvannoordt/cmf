#ifndef CMF_BINARY_FILE_H
#define CMF_BINARY_FILE_H
#include <string>
#include <stdio.h>
#include "CmfError.h"
#include "StringUtils.h"
namespace cmf
{
    
    namespace fmode
    {
        enum fmode
        {
            in=0,
            out=1
        };
    }
    ///@brief Provides a wrapper for binary file pointer
    ///@author WVN
    class BinaryFile
    {
        public:
            ///@brief Constructor
            ///@param filename_in The file to open
            ///@param mode_in The file mode (read/write)
            ///@author WVN
            BinaryFile(std::string filename_in, fmode::fmode mode_in)
            {
                isOpen = false;
                std::string outmode = "w+b";
                mode = mode_in;
                if (mode_in == fmode::in) outmode = "r+b";
                fptr = fopen(filename_in.c_str(), outmode.c_str());
                if (fptr==NULL) { CmfError(strformat("Failed to open file \"{}\"", filename_in)); }
                else { isOpen = true; }
            }
            
            ///@brief Destructor
            ///@author WVN
            ~BinaryFile(void)
            {
                if (isOpen) fclose(fptr);
            }
            
            ///@brief Writes data to the file pointer
            ///@brief rhs Data to write
            ///@author WVN
            template <class streamable> BinaryFile& operator << (const streamable& rhs)
            {
                size_t rs = fwrite((void*)(&rhs), sizeof(streamable), 1, fptr);
                return *this;
            }
            
            ///@brief Reads data from the file pointer
            ///@brief rhs Data to read
            ///@author WVN
            template <class streamable> BinaryFile& operator >> (const streamable& rhs)
            {
                size_t rs = fread((void*)(&rhs), sizeof(streamable), 1, fptr);
                return *this;
            }
            
        private:
            
            ///@brief File pointer
            FILE* fptr;
            
            ///@brief Indicates if the file is open
            bool isOpen;
            
            ///@brief The mode of the open file (input/output)
            fmode::fmode mode;
    };
}

#endif