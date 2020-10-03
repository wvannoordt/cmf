#ifndef CMFOUTPUTSTREAM_H
#define CMFOUTPUTSTREAM_H
#include <vector>
#include <ostream>
#include <iostream>
#include <fstream>
namespace cmf
{
    class cmfoutputflush
    {
        public:
            friend std::ostream & operator << (std::ostream &out, const cmfoutputflush &c) {out << std::endl; return out;}
    };
    extern cmfoutputflush cmfendl;
    class CmfOutputStream
    {
        public:
            CmfOutputStream(void);
            ~CmfOutputStream(void);
            void AddFileToStream(std::string filename);
            CmfOutputStream& operator << (std::string a);
            CmfOutputStream& operator << (double a);
            CmfOutputStream& operator << (const char* a);
            CmfOutputStream& operator << (char* a);
            CmfOutputStream& operator << (int a);
            CmfOutputStream& operator << (cmfoutputflush a);
            CmfOutputStream& operator << (float a);
            CmfOutputStream& operator << (bool a);
        private:
            std::vector<std::ostream*> streams;
            std::vector<std::filebuf*> filebufs;
    };
    extern CmfOutputStream cmfout;
}
#endif
