#include "CmfOutputStream.h"
namespace cmf
{
    cmfoutputflush cmfendl;
    CmfOutputStream cmfout;
    CmfOutputStream::CmfOutputStream(void)
    {
        streams.push_back(&std::cout);
    }
    CmfOutputStream::~CmfOutputStream(void)
    {
        for (size_t i = 0; i < filebufs.size(); i++)
        {
            filebufs[i]->close();
            delete filebufs[i];
        }
        for (size_t i = 1; i < streams.size(); i++)
        {
            delete streams[i];
        }
    }
    void CmfOutputStream::AddFileToStream(std::string filename)
    {
        std::filebuf* next = new std::filebuf();
        filebufs.push_back(next);
        next->open(filename.c_str(), std::ios::out);
        streams.push_back(new std::ostream(next));
    }
}
