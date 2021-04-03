#include "ParallelFile.h"

namespace cmf
{
    ParallelFile::ParallelFile(void)
    {
        filename = "";
        group = &globalGroup;
        isOpen = false;
    }
    
    ParallelFile::ParallelFile(ParallelGroup* group_in)
    {
        filename = "";
        group = group_in;
        isOpen = false;
    }
    
    void ParallelFile::Open(std::string filename_in)
    {
        filename = filename_in;
        isOpen = true;
    }
    
    void ParallelFile::Close(void)
    {
        isOpen = false;
    }
    
    ParallelFile::~ParallelFile(void)
    {
        if (isOpen)
        {
            this->Close();
        }
    }
}