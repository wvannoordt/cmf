#ifndef CMF_SVG_FILE_STREAM_H
#define CMF_SVG_FILE_STREAM_H
#include "CmfError.h"
#include <string>
#include <ostream>
namespace cmf
{
    /// @brief Class that represents a stream to an svg file
	/// @author WVN
    class SvgFileStream
    {
        public:
            /// @brief Constructor
        	/// @author WVN
            SvgFileStream(void)
            {
                isOpen = false;
                openFile = "[none]";
                indentationLevel = 0;
                indentationLevelStr = "";
            }
            
            /// @brief Opens the given file
            /// @param filename the name of the file to open
        	/// @author WVN
            void Open(std::string filename)
            {
                openFile = filename;
                if (isOpen) CmfError("Attempted to open filename \"" + openFile + "\", but the stream is already open");
                isOpen = true;
                myfile.open(filename.c_str());
            }
            
            /// @brief Adds a line to the file with an endline
            /// @param message Writes a line to the open file
        	/// @author WVN
            void AddLine(std::string message)
            {
                if (!isOpen) CmfError("Attempted to write to filename \"" + openFile + "\", but the stream is not open");
                myfile << indentationLevelStr << message << std::endl;
            }
            
            /// @brief Adds a line to the file without an endline
            /// @param message Writes a line to the open file
        	/// @author WVN
            void Add(std::string message)
            {
                if (!isOpen) CmfError("Attempted to write to filename \"" + openFile + "\", but the stream is not open");
                myfile << indentationLevelStr << message;
            }
            
            /// @brief Closes the stream
        	/// @author WVN
            void Close(void)
            {
                if (!isOpen) CmfError("Attempted to close filename \"" + openFile + "\", but the stream is not open");
                myfile.close();
                isOpen = false;
            }
            
            /// @brief Increments the indentation level and returns it
        	/// @author WVN
            int Indent(void)
            {
                indentationLevel++;
                indentationLevelStr = indentationLevelStr + "    ";
                return indentationLevel;
            }
            
            /// @brief Decrements the indentation level and returns it
        	/// @author WVN
            int UnIndent(void)
            {
                if (indentationLevel>0)
                {
                    indentationLevel--;
                    indentationLevelStr = indentationLevelStr.substr(0, indentationLevelStr.length()-4);
                }
                return indentationLevel;
            }
            
            /// @brief Destructor
        	/// @author WVN
            ~SvgFileStream(void)
            {
                if (isOpen) CmfError("Attempted to dispose of filename \"" + openFile + "\", but the stream is not closed");
            }
        
        private:
            
            /// @brief The current indentation level
            int indentationLevel;
            
            /// @brief The current indentation level as a string
            std::string indentationLevelStr;
            
            /// @brief Indicates whether or not the underlying stream is open
            bool isOpen;
            
            /// @brief The underlying stream
            std::ofstream myfile;
            
            /// @brief The name of the most recently opened file
            std::string openFile;
    };
}
#endif