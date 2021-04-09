#ifndef CMF_PARALLEL_FILE_H
#define CMF_PARALLEL_FILE_H
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "CmfPrint.h"
#include "ParallelGroup.h"
#include "ParallelTypes.h"
#include "ParallelDataBuffer.h"
namespace cmf
{
    ///@brief A class that is used for parallel file input and output.
    ///Note that this implementation is slow for large chunks of ascii data.
    ///For Serial writes, only one rank is allowed to write to the file at
    ///a time, and this ranks can be set
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
            
            ///@brief Writes serial data to the ASCII stream if the current rank relative to group is serialRank
            ///@param ts the data to write (streamable)
            ///@author WVN
            template <typename... Ts> void SerialWrite(Ts... ts)
            {
                if (!isOpen) CmfError("Attempted call to ParallelFile::SerialWrite, but there is no file open");
                if (group->Rank()==serialRank)
                {
                    PrintToStream(asciiStream, ts...);
                }
            }
            
            ///@brief Sets the rank (relative to group) that is allowed to write data in serial
            ///@param serialRank_in The serial rank to set
            ///@author WVN
            void SetSerialRank(int serialRank_in);
            
            ///@brief Sets the rank (relative to group) that is allowed to write data in serial
            ///@param newPosition the new position of the file pointer relative to the original position
            ///@author WVN
            void SetPosition(size_t newPosition);
            
            ///@brief Writes a parallel data buffer to the file
            ///@param dataBuffer the parallel data buffer to write
            ///@author WVN
            void ParallelWrite(ParallelDataBuffer& dataBuffer);
            
        private:
            
            ///@brief Performs a parallel write, but does not increment the file pointer
            ///@param offset The offset from the current position in the file
            ///@param pointer The data to write
            ///@param writeSize The size of the data to write
            ///@author WVN
            void ParallelWriteAt(size_t offset, void* pointer, size_t writeSize);
            
            ///@brief Writes the contents of asciiStream to the file
            ///@author WVN
            void PurgeAsciiStream(void);
            
            ///@brief String stream for ascii data writing
            std::ostringstream asciiStream;
            
            ///@brief The filename for this file handle
            std::string filename;
            
            ///@brief The parallel group that this file will be output by
            ParallelGroup* group;
            
            ///@brief The rank that is allowed to write serial information to this file
            int serialRank;
            
            ///@brief Indicates whether or nor the file is open
            bool isOpen;
            
            ///@brief Indicates whether or not all ranks will write provided data to the file
            bool isInParallelMode;
            
            ///@brief Indicates whether or not the data will be written in binary format or ascii format
            bool isInBinaryMode;
            
            ///@brief The current offset relative to the start of the current file
            size_t position;
            
            ///@brief Parallel file handle
            CmfMpiFileHandle parFileHandle;
            
            ///@brief file status handle
            CmfMpiFileStatus parFileStatus;
    };
}

#endif