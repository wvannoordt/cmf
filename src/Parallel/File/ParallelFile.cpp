#include "ParallelFile.h"
#include "CmfPrint.h"

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
        asciiStream = std::ostringstream();
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
    
    void ParallelFile::Open(std::string filename_in)
    {
#if(!CMF_PARALLEL)
        CmfError("ParallelFile::Open() not yet supported with cerial CMF build");
#endif
        position = 0;
        filename = filename_in;
        isOpen = true;
        group->Synchronize();

#if (CMF_PARALLEL)

            CMF_MPI_CHECK(MPI_File_open(group->GetComm(), filename.c_str(), MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &parFileHandle));
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

    void ParallelFile::ParallelWriteAt(size_t offset, void* pointer, size_t writeSize)
    {
        if (!isOpen) CmfError("Attempted call to ParallelFile::ParallelWriteAt, but there is no file open");
#if (CMF_PARALLEL)
        CMF_MPI_CHECK(MPI_File_write_at(parFileHandle, position+offset, pointer, writeSize, MPI_CHAR, &parFileStatus));
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