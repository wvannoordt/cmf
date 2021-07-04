#ifndef CMFOUTPUTSTREAM_H
#define CMFOUTPUTSTREAM_H
#include <vector>
#include <ostream>
#include <iostream>
#include <fstream>
#include <sstream>
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
    
    /// @brief Interface for attaching an endpoint to the CMF stream
	/// @author WVN
    class ICmfOutputStreamEndpoint
    {
        public:
            /// @brief Adds a string to the endpoint implementation
            /// @param msg the string to add
        	/// @author WVN
            virtual void AddString(const std::string& msg)=0;
    };
    
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
            
            /// @brief Streams a... thing.
            /// @param a The thing to stream
            /// @author WVN
            template <typename T> CmfOutputStream& operator << (T a)
            {
                for (auto& strm: streams) *strm << a;
                if (endpoints.size()>0)
                {
                    std::stringstream ss;
                    ss << a;
                    for (auto& endpt: endpoints) endpt->AddString(ss.str());
                }
                return *this;
            }
            
            /// @brief Clears all endpoints
            /// @author WVN
            void ClearEndpoints(void)
            {
                endpoints.clear();
            }
            
            /// @brief Adds a generic endpoint to the stream
            /// @param endpt The endpoint to add
            /// @author WVN
            void AddEndpoint(ICmfOutputStreamEndpoint* endpt)
            {
                endpoints.push_back(endpt);
            }
            
        private:
                        
            /// @brief List of streams to be output to
            std::vector<std::ostream*> streams;
            /// @brief List of file buffers that the streams output to
            std::vector<std::filebuf*> filebufs;
            /// @brief List of additional stream endpoints
            std::vector<ICmfOutputStreamEndpoint*> endpoints;
    };
    extern CmfOutputStream cmfout;
}
#endif
