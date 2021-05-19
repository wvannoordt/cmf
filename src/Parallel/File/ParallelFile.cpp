#include "ParallelFile.h"
#include "CmfPrint.h"
#include "StringUtils.h"
#include "Utils.hx"

namespace cmf
{
    ParallelFile::ParallelFile(void)
    {
        filename = "";
        group = &globalGroup;
        isOpen = false;
        isInParallelMode = false;
        isInBinaryMode = false;
        position = 0;
        serialRank = 0;
        endLine = "\n";
        openFileSize = 0;
    }
    
    ParallelFile::ParallelFile(ParallelGroup* group_in)
    {
        filename = "";
        group = group_in;
        isOpen = false;
        isInParallelMode = false;
        isInBinaryMode = false;
        position = 0;
        serialRank = 0;
        endLine = "\n";
        openFileSize = 0;
    }

    void ParallelFile::PurgeAsciiStream(void)
    {
        size_t writeSize = 0;
        if (group->Rank()==serialRank)
        {
            std::string strn = asciiStream.str();
            writeSize = strn.length();
            ParallelWriteAt(0, (void*)strn.c_str(), writeSize);
        }
        size_t globalWriteSize = group->Sum(writeSize);
        SetPosition(position + globalWriteSize);
        asciiStream.str("");
        asciiStream.clear();
    }
    
    void ParallelFile::SetPosition(size_t newPosition)
    {
        position = newPosition;
    }
    
    void ParallelFile::SetSerialRank(int serialRank_in)
    {
        this->PurgeAsciiStream();
        serialRank = serialRank_in;
    }
    
    size_t ParallelFile::SeekString(std::string token)
    {
        bool foundToken = false;
        size_t maxAllowableSearchSize = CMFMIN(maxLineLength, openFileSize);
        //Arbitrary choice of size for now, how to set this?
        size_t bufferSize = 1400;
        size_t tokenLength = token.length();
        std::vector<char> sBuf;
        sBuf.resize(bufferSize+1, 0);
        std::fill(sBuf.begin(), sBuf.end(), 0);
        size_t currentOffset = 0;
        size_t lastOffset = 0;
        size_t tokenOffsetRelativeToBuffer = 0;
        while (!foundToken && ((position + currentOffset) <= maxAllowableSearchSize))
        {
            lastOffset = currentOffset;
            size_t readSize = CMFMIN(bufferSize, openFileSize-(position+currentOffset));
            ParallelReadAt(currentOffset, &sBuf[0], readSize);
            std::string line(sBuf.begin(), sBuf.begin()+readSize);
            tokenOffsetRelativeToBuffer = line.find(token);
            foundToken = (tokenOffsetRelativeToBuffer!=std::string::npos);
            currentOffset += (readSize-tokenLength);            
        }
        if (!foundToken) return std::string::npos;
        size_t output = lastOffset + tokenOffsetRelativeToBuffer;
        return output;
    }
    
    bool ParallelFile::AdvanceTo(std::string token)
    {
        size_t advanceSize = SeekString(token);
        if (advanceSize==std::string::npos) return false;
        SetPosition(position + advanceSize);
        return true;
    }
    
    bool ParallelFile::AdvanceBeyond(std::string token)
    {
        size_t advanceSize = SeekString(token);
        if (advanceSize==std::string::npos) return false;
        SetPosition(position + advanceSize + token.length());
        return true;
    }
    
    std::string ParallelFile::SerialRead(void)
    {
        std::string output = this->Read();
        if (group->Rank() == serialRank) return output;
        return "";
    }

    std::string ParallelFile::Read(void)
    {
        size_t lineSize = SeekString(this->endLine);
        if (lineSize==std::string::npos) return "";
        if (lineSize > this->maxLineLength)
        {
            CmfError(strformat("Attempted to read a huge line of size {}, larger than max line length {}, from file {}", lineSize, this->maxLineLength, filename));
        }
        std::vector<char> output;
        output.resize(lineSize, 0);
        ParallelReadAt(0, &output[0], lineSize);
        this->SetPosition(position+lineSize+1);
        std::string outputStr(output.begin(), output.end());
        return outputStr;
    }
    
    void ParallelFile::Open(std::string filename_in)
    {
#if(!CMF_PARALLEL)
        CmfError("ParallelFile::Open() not yet supported with serial CMF build");
#endif
        position = 0;
        filename = filename_in;
        isOpen = true;
        group->Synchronize();

#if (CMF_PARALLEL)

            CMF_MPI_CHECK(MPI_File_open(group->GetComm(), filename.c_str(), MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &parFileHandle));
            MPI_Offset mofst;
            CMF_MPI_CHECK(MPI_File_get_size(parFileHandle, &mofst));
            openFileSize = (size_t)mofst;
            SetPosition(0);

#endif
    }
    
    void ParallelFile::ParallelWrite(ParallelDataBuffer& dataBuffer)
    {
        PurgeAsciiStream();
        group->Synchronize();
        size_t numBufs = dataBuffer.pointers.size();
        size_t totalWriteSize = 0;
        for (auto& s:dataBuffer.sizes) totalWriteSize += s;
        size_t totalWriteSizeGlob = group->Sum(totalWriteSize);
        for (size_t i = 0; i < numBufs; i++)
        {
            ParallelWriteAt(dataBuffer.offsets[i], dataBuffer.pointers[i], dataBuffer.sizes[i]);
        }
        SetPosition(position + totalWriteSizeGlob);
        group->Synchronize();
    }
    
    void ParallelFile::ParallelRead(ParallelDataBuffer& dataBuffer)
    {
        group->Synchronize();
        size_t numBufs = dataBuffer.pointers.size();
        size_t totalWriteSize = 0;
        for (auto& s:dataBuffer.sizes) totalWriteSize += s;
        size_t totalWriteSizeGlob = group->Sum(totalWriteSize);
        for (size_t i = 0; i < numBufs; i++)
        {
            ParallelReadAt(dataBuffer.offsets[i], dataBuffer.pointers[i], dataBuffer.sizes[i]);
        }
        SetPosition(position + totalWriteSizeGlob);
        group->Synchronize();
    }

    void ParallelFile::ParallelWriteAt(size_t offset, void* pointer, size_t writeSize)
    {
        if (!isOpen) CmfError("Attempted call to ParallelFile::ParallelWriteAt, but there is no file open");
#if (CMF_PARALLEL)
        CMF_MPI_CHECK(MPI_File_write_at(parFileHandle, position+offset, pointer, writeSize, MPI_CHAR, &parFileStatus));
#endif
    }
    
    void ParallelFile::ParallelReadAt(size_t offset, void* pointer, size_t readSize)
    {
        if (!isOpen) CmfError("Attempted call to ParallelFile::ParallelWriteAt, but there is no file open");
#if (CMF_PARALLEL)
        CMF_MPI_CHECK(MPI_File_read_at(parFileHandle, position+offset, pointer, readSize, MPI_CHAR, &parFileStatus));
#endif
    }
    
    void ParallelFile::Close(void)
    {
        group->Synchronize();
        if (isOpen)
        {
            PurgeAsciiStream();
            isOpen = false;

#if (CMF_PARALLEL)

            CMF_MPI_CHECK(MPI_File_close(&parFileHandle));

#endif

        }
    }
    
    ParallelFile::~ParallelFile(void)
    {
        if (isOpen)
        {
            this->Close();
        }
    }
}