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
                if (!isOpen) CmfError("Attempted call to ParallelFile::SerialWriteLine, but there is no file open");
                if (group->Rank().id==serialRank)
                {
                    PrintToStream(asciiStream, ts...);
                }
            }
            
            ///@brief Writes a line to the ASCII stream
            ///@param ts the data to write (streamable)
            ///@author WVN
            template <typename... Ts> void Write(Ts... ts)
            {
                int curRank = this->serialRank;
                this->SetSerialRank(0);
                if (!isOpen) CmfError("Attempted call to ParallelFile::WriteLine, but there is no file open");
                SerialWrite(ts...);
                group->Synchronize();
            }

            ///@brief Reads a line from the file to only the current serialRank. All ranks must call this function as it is blocking.
            ///Retuns an empty string if the current rank is not the serialRank
            ///@author WVN
            std::string SerialRead(void);
            
            ///@brief Reads a line from the file to all ranks. All ranks must call this function as it is blocking
            ///@author WVN
            std::string Read(void);
            
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
            
            ///@brief Read a parallel data buffer from the file
            ///@param dataBuffer the parallel data buffer to write
            ///@author WVN
            void ParallelRead(ParallelDataBuffer& dataBuffer);
            
            ///@brief Sets the file offset to the first character of the given string and returns true.
            ///If the given string cannot be found, returns false and does nothing.
            ///@param token The string to search for
            ///@author WVN
            bool AdvanceTo(std::string token);
            
            ///@brief Sets the file offset to the first character after the given string and returns true.
            ///If the given string cannot be found, returns false and does nothing.
            ///@param token The string to search for
            ///@author WVN
            bool AdvanceBeyond(std::string token);
            
            ///@brief Returns the name of the open file
            ///@author WVN
            std::string OpenFileName(void) { return filename;}
            
        private:
            
            
            ///@brief Seeks the provided string and returns the offset from the current poition until the first character
            ///of the first instance of that string, or returns std::string::npos otherwise. This function might be slow.
            ///Note that this will yield a flase negative if the next instance of the token is futher than maxLineLength
            ///away from the current position
            ///@param token The string to search for
            ///@author WVN
            size_t SeekString(std::string token);
            
            ///@brief Performs a parallel write, but does not increment the file pointer
            ///@param offset The offset from the current position in the file
            ///@param pointer The data to write
            ///@param writeSize The size of the data to write
            ///@author WVN
            void ParallelWriteAt(size_t offset, void* pointer, size_t writeSize);
            
            ///@brief Performs a parallel read, but does not increment the file pointer
            ///@param offset The offset from the current position in the file
            ///@param pointer The data to read
            ///@param writeSize The size of the data to read
            ///@author WVN
            void ParallelReadAt(size_t offset, void* pointer, size_t readSize);
            
            ///@brief Writes the contents of asciiStream to the file
            ///@author WVN
            void PurgeAsciiStream(void);
            
            ///@brief String stream for ascii data writing
            std::ostringstream asciiStream;
            
            ///@brief The size of the current open file, only valid for reading
            size_t openFileSize;
            
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
            
            ///@brief The end-of-line string
            std::string endLine;
            
            ///@brief A precautionary maximum line length for a line read
            const size_t maxLineLength = 16777216;
            
            ///@brief Parallel file handle
            CmfMpiFileHandle parFileHandle;
            
            ///@brief file status handle
            CmfMpiFileStatus parFileStatus;
    };
}

#endif