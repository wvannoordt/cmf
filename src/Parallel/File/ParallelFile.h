#ifndef CMF_PARALLEL_FILE_H
#define CMF_PARALLEL_FILE_H
#include <string>
#include <fstream>
#include <sstream>
#include "ParallelGroup.h"
namespace cmf
{
    ///@brief A class that is used for parallel file input and output.
    ///Note that this implementation is slow for large chunks of ascii data
    ///@author WVN
    class ParallelFile
    {
        public:
            ///@brief Constructor
            ///@author WVN
            ParallelFile(void);
            
            ///@brief Constructor
            ///@param group_in The parallel group that this file will be output by
            ///@author WVN
            ParallelFile(ParallelGroup* group_in);
            
            ///@brief Destructor
            ///@author WVN
            ~ParallelFile(void);
            
            ///@brief Opens the file with the provided name
            ///@param filename_in the name of the file
            ///@author WVN
            void Open(std::string filename_in);
            
            ///@brief Closes the open file, if any
            ///@author WVN
            void Close(void);
            
        private:
            
            ///@brief The filename for this file handle
            std::string filename;
            
            ///@brief The parallel group that this file will be output by
            ParallelGroup* group;
            
            ///@brief Indicates whether or nor the file is open
            bool isOpen;
            
            ///@brief Indicates whether or not all ranks will write provided data to the file
            bool isInParallelMode;
            
            ///@brief Indicates whether or not the data will be written in binary format or ascii format
            bool isInBinaryMode;
            
            ///@brief String stream for ascii data writing
            std::ostringstream asciiStream;
    };
}

#endif