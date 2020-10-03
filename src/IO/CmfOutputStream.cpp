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
    CmfOutputStream& CmfOutputStream::operator << (std::string a)
    {
        for (size_t i = 0; i < streams.size(); i++) *streams[i] << a;
        return *this;
    }
    CmfOutputStream& CmfOutputStream::operator << (double a)
    {
        for (size_t i = 0; i < streams.size(); i++) *streams[i] << a;
        return *this;
    }
    CmfOutputStream& CmfOutputStream::operator << (const char* a)
    {
        for (size_t i = 0; i < streams.size(); i++) *streams[i] << a;
        return *this;
    }
    CmfOutputStream& CmfOutputStream::operator << (char* a)
    {
        for (size_t i = 0; i < streams.size(); i++) *streams[i] << a;
        return *this;
    }
    CmfOutputStream& CmfOutputStream::operator << (int a)
    {
        for (size_t i = 0; i < streams.size(); i++) *streams[i] << a;
        return *this;
    }
    CmfOutputStream& CmfOutputStream::operator << (cmfoutputflush a)
    {
        for (size_t i = 0; i < streams.size(); i++) *streams[i] << std::endl;
        return *this;
    }
    CmfOutputStream& CmfOutputStream::operator << (float a)
    {
        for (size_t i = 0; i < streams.size(); i++) *streams[i] << a;
        return *this;
    }
    CmfOutputStream& CmfOutputStream::operator << (bool a)
    {
        for (size_t i = 0; i < streams.size(); i++) *streams[i] << a;
        return *this;
    }
}
