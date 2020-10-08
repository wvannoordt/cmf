#ifndef CMFOUTPUTSTREAM_H
#define CMFOUTPUTSTREAM_H
#include <vector>
#include <ostream>
#include <iostream>
#include <fstream>
namespace cmf
{
    /// @brief Class providing cmfendl for nice syntax.
	/// @author WVN
    class cmfoutputflush
    {
        public:
            friend std::ostream & operator << (std::ostream &out, const cmfoutputflush &c) {out << std::endl; return out;}
    };
    extern cmfoutputflush cmfendl;
    /// @brief Class providing custom output streams for CMF. Generally can be used e.g.
    /// \code{.cpp}
    /// cmfout << "hello world" << cmfendl;
    /// \endcode
	/// @author WVN
    class CmfOutputStream
    {
        public:
            CmfOutputStream(void);
            ~CmfOutputStream(void);
            /// @brief Add a file to the output stream. Used to create log files.
            /// @param filename The name of a file to add
            /// @author WVN
            void AddFileToStream(std::string filename);
            CmfOutputStream& operator << (std::string a);
            CmfOutputStream& operator << (double a);
            CmfOutputStream& operator << (const char* a);
            CmfOutputStream& operator << (char* a);
            CmfOutputStream& operator << (int a);
            CmfOutputStream& operator << (size_t a);
            CmfOutputStream& operator << (cmfoutputflush a);
            CmfOutputStream& operator << (float a);
            CmfOutputStream& operator << (bool a);
        private:
            /// @brief List of streams to be output to
            std::vector<std::ostream*> streams;
            /// @brief List of file buffers that the streams output to
            std::vector<std::filebuf*> filebufs;
    };
    extern CmfOutputStream cmfout;
}
#endif
