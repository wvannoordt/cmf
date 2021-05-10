#ifndef CMF_PATH_H
#define CMF_PATH_H
#include <string>
#include <vector>
#include "CmfError.h"
#include "StringUtils.h"
namespace cmf
{
    /// @brief Represents a path in the host file-system. Intended to be platform-independent
    /// @author WVN
    class Path
    {
        public:
            /// @brief Constructor
            /// @author WVN
            Path(void)
            {
                components.push_back(".");
            }
            
            /// @brief Constructor
            /// @param pathString A string representing the parth to the desired location
            /// @author WVN
            Path(std::string pathString)
            {
                components = StringSplit(pathString, "/");
            }
            
            /// @brief Returns the string that the path represents
            /// @author WVN
            std::string Str(void)
            {
                std::string output = "";
                if (components.size()>0) output += components[0];
                for (int i = 1; i < components.size(); i++)
                {
                    output += "/";
                    output += components[i];
                }
                return output;
            }
            
            ///@brief append operator
            ///@param str the string to append
            ///@author WVN
            Path& operator += (std::string& str)
            {
                Path addPath(str);
                this->Append(addPath);
                return *this;
            }
            
            ///@brief append operator
            ///@param pth the string
            ///@author WVN
            Path& operator += (const char* pth)
            {
                std::string s(pth);
                Path pth1(s);
                this->Append(pth1);
                return *this;
            }
            
            ///@brief append operator
            ///@param pth the path to append
            ///@author WVN
            Path& operator += (Path& pth)
            {
                this->Append(pth);
                return *this;
            }
            
            ///@brief appends a path to this path
            ///@param pth the path to append
            ///@author WVN
            void Append(Path& pth)
            {
                for (auto& s:pth.components) components.push_back(s);
            }
        
        private:
            
            ///@brief A list of individual components of the path
            std::vector<std::string> components;
            
    };
    
    ///@brief Stream operator
    ///@param stream Original stream
    ///@param pth the path to stream
    ///@author WVN
    static std::ostream& operator<< (std::ostream& stream, Path& pth)
    {
        stream << pth.Str();
        return stream;
    }
}

#endif