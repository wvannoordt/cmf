#ifndef VTK_ATTR_H
#define VTK_ATTR_H
#include <iostream>
#include "CmfError.h"
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <map>
#include <cstring>
#include "CmfGC.h"

#define CMF_VTK_BUF_SIZE 4096
#define CMF_VTK_MAX_STRING_SIZE 256

namespace cmf
{
    namespace VtkAttributableType
    {
        enum VtkAttributableType
        {
            stringType, doubleType, intType, longType
        };
    }

    /// @brief Stores the data associated with an attribute
    /// @author WVN
    struct VtkAttribute
    {
        /// @brief Name of the attribute
        std::string name;
        
        /// @brief Type of the attribute
        VtkAttributableType::VtkAttributableType attrType;
        
        /// @brief Location of the attribute
        size_t attrOffset;
        
        /// @brief Size of the attribute
        size_t varSize;
    };

    /// @brief Class (to be inherited) providing a way to get/set generic attributes
    /// @author WVN
    class VtkAttributable
    {
        public:
            /// @brief Set the name of the attribute that represents the buffer size
            /// @param bufferSizeName The name of the attribute corresponding to the buffer size
            /// @author WVN
            void BufferSizeIs(std::string bufferSizeName)
            {
                bufferSizeAttrName = bufferSizeName;
            }

            /// @brief Set the name of the attribute that represents the stride
            /// @param strideName_in The name of the attribute corresponding to the buffer stride
            /// @author WVN
            void StrideIs(std::string strideName_in)
            {
                strideName = strideName_in;
            }

            /// @brief Returns true if the provided attrubte name is a required attrubte
            /// @param name The name of the attribute to check
            /// @author WVN
            bool RequiredAttributeExists(std::string name)
            {
                std::map<std::string,VtkAttributableType::VtkAttributableType>::iterator it = requiredAttributes.find(name);
                if(it != requiredAttributes.end())
                {
                   return true;
                }
                return false;
            }

            /// @brief Returns true if the provided attrubte name exists within the provided attributes
            /// @param name The name of the attribute to check
            /// @author WVN
            bool AttributeExists(std::string name)
            {
                std::map<std::string,VtkAttribute>::iterator it = attributes.find(name);
                if(it != attributes.end())
                {
                   return true;
                }
                return false;
            }

            /// @brief Asserts that all required attributes are found
            /// @pre Throws an error if not all required attributes are found
            /// @author WVN
            void EnforceAllRequiredAttributes(void)
            {
                for (std::map<std::string,VtkAttributableType::VtkAttributableType>::iterator it = requiredAttributes.begin(); it != requiredAttributes.end(); it++)
                {
                    if (!AttributeExists(it->first)) CmfError("Cannot find required attribute \"" + it->first + "\" for object \"" + className + "\".");
                }
            }
            
            /// @brief Adds a required attribute
            /// @param name The name of the required attribute to add
            /// @param attribType The expected type of the required attribute
            /// \see VtkAttributableType
            /// @author WVN
            void AddRequiredAttribute(std::string name, VtkAttributableType::VtkAttributableType attribType)
            {
                if (RequiredAttributeExists(name)) CmfError("Multiple definition of attribute " + name);
                requiredAttributes.insert({name, attribType});
            }

            /// @brief Provides a string representation of a VtkAttributableType
            /// @param typein The VtkAttributableType to provide a string representation of
            /// \see VtkAttributableType
            /// @author WVN
            std::string AttrTypeStr(VtkAttributableType::VtkAttributableType typein)
            {
                switch (typein)
                {
                    case VtkAttributableType::stringType: return "string";
                    case VtkAttributableType::doubleType: return "double";
                    case VtkAttributableType::intType:    return "int";
                    case VtkAttributableType::longType:   return "long";
                }
            }

            /// @brief Creates a new VtkAttribute.
            /// @param newAttrValues A VtkAttribute object to be created
            /// @param name The name of the new attribute
            /// @param data The location of data to copy (copied to dataBuffer) pertaining to this attribute
            /// @param setType The type of the new attribute
            /// \see VtkAttributableType
            /// @author WVN
            void CreateNewAttribute(VtkAttribute* newAttrValues, std::string name, void* data, VtkAttributableType::VtkAttributableType setType)
            {
                newAttrValues->name = name;
                newAttrValues->attrType = setType;
                newAttrValues->attrOffset = nextPointer;
                size_t storedSize = 0;
                char* cPtr;
                switch (setType)
                {
                    case VtkAttributableType::intType:
                    {
                        cPtr = (char*)data;
                        storedSize = sizeof(int);
                        break;
                    }
                    case VtkAttributableType::doubleType:
                    {
                        cPtr = (char*)data;
                        storedSize = sizeof(double);
                        break;
                    }
                    case VtkAttributableType::longType:
                    {
                        cPtr = (char*)data;
                        storedSize = sizeof(size_t);
                        break;
                    }
                    case VtkAttributableType::stringType:
                    {
                        std::string str = *((std::string*)data);
                        storedSize = str.length() + 1;
                        cPtr = (char*)(str.c_str());
                        break;
                    }
                }
                newAttrValues->varSize = storedSize;
                memcpy(attributeBuffer+nextPointer, cPtr, storedSize);
                nextPointer += storedSize;
            }

            /// @brief Returns true if the provided types can be cast to each other according to the current implementation
            /// @param a A VtkAttributableType to be checked against b
            /// @param b A VtkAttributableType to be checked against a
            /// \see VtkAttributableType
            /// @author WVN
            bool TypesAreCompatible(VtkAttributableType::VtkAttributableType a, VtkAttributableType::VtkAttributableType b)
            {
                if ((a==VtkAttributableType::intType) && (b==VtkAttributableType::longType)) return true;
                if ((b==VtkAttributableType::intType) && (a==VtkAttributableType::longType)) return true;
                if ((b==VtkAttributableType::longType) && (a==VtkAttributableType::longType)) return true;
                if ((b==VtkAttributableType::intType) && (a==VtkAttributableType::intType)) return true;
                return false;
            }

            /// @brief Identical to the sizeof operator, but for the VtkAttributableType of the current buffer
            /// @param a A VtkAttributableType to be checked against b
            /// @param b A VtkAttributableType to be checked against a
            /// \see VtkAttributableType
            /// @author WVN
            size_t GetBufferElementSize(void)
            {
                switch (bufferType)
                {
                    case VtkAttributableType::intType: return sizeof(int);
                    case VtkAttributableType::doubleType: return sizeof(double);
                    case VtkAttributableType::longType: return sizeof(size_t);
                    case VtkAttributableType::stringType: CmfError("Error: bufferType is set to incompatible type (" + AttrTypeStr(bufferType) + ").");
                }
            }

            /// @brief Sets a new attribute and parses the provided data.
            /// @param name The name of the new attribute
            /// @param data The location of data to copy (copied to dataBuffer) pertaining to this attribute
            /// @param setType The type of the new attribute
            /// \see VtkAttributableType
            /// @author WVN
            void SetAttribute(std::string name, void* data, VtkAttributableType::VtkAttributableType setType)
            {
                if (!RequiredAttributeExists(name))
                {
                    std::cout << "Error: Attempt to set nonexistent attribute \"" << name << "\"." << std::endl;
                    std::cout << "Existing attribute names are:" << std::endl;
                    for (std::map<std::string,VtkAttributableType::VtkAttributableType>::iterator it = requiredAttributes.begin(); it != requiredAttributes.end(); it++)
                    {
                        std::cout << " => " << it->first << " (" << AttrTypeStr(it->second) << ")" << std::endl;
                    }
                    CmfError("Stopping");
                }
                if ((requiredAttributes[name] != setType) && (!TypesAreCompatible(requiredAttributes[name], setType)))
                {
                    std::cout << "Error: Attempt to set attribute \"" << name << "\" to type " << std::endl;
                    std::cout << "\"" << AttrTypeStr(setType) << "\" but it is of type \"" << AttrTypeStr(requiredAttributes[name]) << "\"" << std::endl;
                    CmfError("Stopping");
                }
                VtkAttribute newAttrValues;
                CreateNewAttribute(&newAttrValues, name, data, setType);
                attributes.insert({name, newAttrValues});
                if (name == bufferSizeAttrName)
                {
                    if (!TypesAreCompatible(setType, VtkAttributableType::longType))
                    {
                        CmfError("Attempted to allocate dataBuffer using attribute \"" + name + "\" of improper type " + AttrTypeStr(setType) + ".");
                    }
                    size_t allocsize = ((setType == VtkAttributableType::intType) ? (*((int*)data)) : (*((size_t*)data))) * GetBufferElementSize();
                    //VTK buffers are all temporary: disallow stack allocation
                    SetStackAllocationAllowed(false);
                    dataBuffer = (char*)Cmf_Alloc(allocsize);
                    SetStackAllocationAllowed(true);
                    bufferIsAllocated = true;
                    allocatedSize = allocsize;
                }
                if (name == strideName)
                {
                    if (!TypesAreCompatible(setType, VtkAttributableType::longType))
                    {
                        CmfError("Attempted to set stride using attribute \"" + name + "\" of improper type " + AttrTypeStr(setType) + ".");
                    }
                    stride = (int)((setType == VtkAttributableType::intType) ? (*((int*)data)) : (*((size_t*)data)));
                }
            }

            /// @brief Expose the underlying buffer.
            /// @param bufOut Location to place the address of the underlying buffer
            /// @param outputSize Set to the size of the underlying buffer
            /// @author WVN
            void GetBuffer(int**    bufOut, size_t* outputSize) {*outputSize = allocatedSize; *bufOut = (int*)    dataBuffer;}
            
            /// @brief Expose the underlying buffer.
            /// @param bufOut Location to place the address of the underlying buffer
            /// @author WVN
            void GetBuffer(int**    bufOut)                     {                             *bufOut = (int*)    dataBuffer;}
            
            /// @brief Expose the underlying buffer.
            /// @param bufOut Location to place the address of the underlying buffer
            /// @param outputSize Set to the size of the underlying buffer
            /// @author WVN
            void GetBuffer(double** bufOut, size_t* outputSize) {*outputSize = allocatedSize; *bufOut = (double*) dataBuffer;}
            
            /// @brief Expose the underlying buffer.
            /// @param bufOut Location to place the address of the underlying buffer
            /// @author WVN
            void GetBuffer(double** bufOut)                     {                             *bufOut = (double*) dataBuffer;}

            /// @brief Set attribute data by name
            /// @param name The name of the attribute to set
            /// @param data The data associated with the attribute name
            /// @author WVN
            void SetAttribute(std::string name, int data)         {SetAttribute(name, (void*)&data, VtkAttributableType::intType);   }
            
            /// @brief Set attribute data by name
            /// @param name The name of the attribute to set
            /// @param data The data associated with the attribute name
            /// @author WVN
            void SetAttribute(std::string name, double data)      {SetAttribute(name, (void*)&data, VtkAttributableType::doubleType);}
            
            /// @brief Set attribute data by name
            /// @param name The name of the attribute to set
            /// @param data The data associated with the attribute name
            /// @author WVN
            void SetAttribute(std::string name, size_t data)      {SetAttribute(name, (void*)&data, VtkAttributableType::longType);  }
            
            /// @brief Set attribute data by name
            /// @param name The name of the attribute to set
            /// @param data The data associated with the attribute name
            /// @author WVN
            void SetAttribute(std::string name, std::string data) {SetAttribute(name, (void*)&data, VtkAttributableType::stringType);}

            /// @brief Frees underlying buffer
            /// @author WVN
            void Destroy(void)
            {
                if (bufferIsAllocated)
                {
                    bufferIsAllocated = false;
                    Cmf_Free(dataBuffer);
                }
            }

            /// @brief Set the attribute format string for the current object
            /// @param format_in The format string of the current object
            /// @author WVN
            void SetFormat(std::string format_in)
            {
                format = format_in;
            }
            
            /// @brief Set the attribute format string for the current object
            /// @param name_in The name of the current VtkAttributable
            /// @param namatrTypee_in The type of the current VtkAttributable
            /// \see VtkAttributableType
            /// @author WVN
            VtkAttributable(std::string name_in, VtkAttributableType::VtkAttributableType atrType)
            {
                bufferType = atrType;
                bufferIsAllocated = false;
                nextPointer = 0;
                className = name_in;
                allocatedSize = 0;
                stride = -1;
                BufferSizeIs("bufferCount");
                StrideIs("stride");
                format = "(NO FORMAT SPECIFIED)";
                invocationStart = "${";
                invocationEnd = "}";
            }

            /// @brief The destructor for VtkAttributable, calls Destroy
            /// \see Destroy
            /// @author WVN
            ~VtkAttributable(void)
            {
                Destroy();
            }

            /// @brief Checks if the end of a string is at a given position
            /// @param i The position in question
            /// @param str The string to find
            /// @author WVN
            bool PositionIsStart(size_t i, std::string str)
            {
                if (i < invocationStart.length()-1) return false;
                return (str.substr(i-invocationStart.length()+1, invocationStart.length())==invocationStart);
            }

            /// @brief Checks if the beginning of a string is at a given position
            /// @param i The position in question
            /// @param str The string to find
            /// @author WVN
            bool PositionIsEnd(size_t i, std::string str)
            {
                if (i+invocationEnd.length()-1 >= (str.length())) return false;
                return (str.substr(i, invocationEnd.length()) == invocationEnd);
            }

            /// @brief Checks if the given string has bracket consistency
            /// @param str The string to test
            /// @author WVN
            void AssertBracketConsistency(std::string str)
            {
                int level = 0;
                for (size_t i = 0; i < str.length(); i++)
                {
                    if (PositionIsStart(i, str)) level++;
                    if (PositionIsEnd(i, str))
                    {
                        level--;
                        if (level<0) CmfError("Invocation \"" + str + "\" has inconsistent brackets.");
                    }
                }
                if (level!=0) CmfError("Invocation \"" + str + "\" has inconsistent brackets.");
            }

            /// @brief Returns the string representation of the attribute
            /// @param keyValue The name of the attribute to retrieve
            /// @author WVN
            std::string GetDefinition(std::string keyValue)
            {
                if (!AttributeExists(keyValue)) CmfError("Cannot find attribute \"" + keyValue + "\", requested in format of attribute \"" + className + "\"");
                char buffer[CMF_VTK_MAX_STRING_SIZE] = {0};
                VtkAttribute info = attributes[keyValue];
                memcpy(buffer, attributeBuffer + info.attrOffset, info.varSize);
                switch (info.attrType)
                {
                    case VtkAttributableType::intType:
                    {
                        int output = *((int*)buffer);
                        return std::to_string(output);
                    }
                    case VtkAttributableType::doubleType:
                    {
                        double output = *((double*)buffer);
                        return std::to_string(output);
                    }
                    case VtkAttributableType::longType:
                    {
                        size_t output = *((size_t*)buffer);
                        return std::to_string(output);
                    }
                    case VtkAttributableType::stringType:
                    {
                        std::string output(buffer);
                        return output;
                    }
                }
            }

            /// @brief Recursively resolves a string within a preprocessor expansion context
            /// @param str The string to resolve
            /// @param level The number of current recursions
            /// @author WVN
            std::string HeaderString(std::string str, int level)
            {
                if (str.length()==0) return str;
                AssertBracketConsistency(str);
                size_t start, end;
                start = str.find(invocationStart);
                if (start==std::string::npos)
                {
                    return str;
                }
                int bracketLevel = 1;
                for (size_t i = start+invocationStart.length(); i < str.length(); i++)
                {
                    if (PositionIsStart(i, str)) bracketLevel++;
                    if (PositionIsEnd(i, str)) bracketLevel--;
                    if (bracketLevel==0) {end = i; break;}
                }
                std::string pre = str.substr(0, start);
                std::string med = str.substr(start+invocationStart.length(), end-start-invocationStart.length());
                std::string post = str.substr(end+invocationEnd.length(), str.length()-end-invocationEnd.length());
                std::string output = pre + GetDefinition(HeaderString(med, level+1)) + HeaderString(post, level+1);
                return output;
            }

            /// @brief Recursively resolves a string within a preprocessor expansion context
            /// @param str The string to resolve
            /// @author WVN
            std::string HeaderString(void) {return HeaderString(format, 0);}

            /// @brief Writes generic data to a file stream
            /// @param myfile The file stream to be written to
            /// @author WVN
            template <typename T> void WriteAs(std::ofstream & myfile)
            {
                size_t elemSize = GetBufferElementSize();
                size_t totalEntries = allocatedSize/elemSize;
                for (size_t idx = 0; idx < totalEntries; idx++)
                {
                    if ((idx%stride==0)&&(idx>0)) myfile << std::endl;
                    myfile << *((T*)(dataBuffer+idx*elemSize));
                    if (((idx+1)%stride!=0)) myfile << " ";
                }
            }

            /// @brief Writes the current object to a file stream
            /// @param myfile The file stream to be written to
            /// @author WVN
            void Write(std::ofstream & myfile)
            {
                myfile << HeaderString() << std::endl;
                switch (bufferType)
                {
                    case VtkAttributableType::intType:
                    {
                        WriteAs<int>(myfile);
                        break;
                    }
                    case VtkAttributableType::longType:
                    {
                        WriteAs<size_t>(myfile);
                        break;
                    }
                    case VtkAttributableType::doubleType:
                    {
                        WriteAs<double>(myfile);
                        break;
                    }
                }
                myfile << std::endl;
            }

        protected:
            /// @brief Stores all attribute data
            char attributeBuffer[CMF_VTK_BUF_SIZE] = {0};
            
            /// @brief Index of current storage location
            int nextPointer;
            
            /// @brief A list of required attributes by name
            std::map<std::string, VtkAttributableType::VtkAttributableType> requiredAttributes;
            
            /// @brief A list of existing attributes by name
            std::map<std::string, VtkAttribute> attributes;
            
            /// @brief Retrieves the underlying data buffer
            /// @param buf An address to copy the underlying buffer pointer to
            /// @param outputSize An address to copy the underlying buffer size to
            /// @author WVN
            void GetBuffer(char** buf, size_t* outputSize)
            {
                if (!bufferIsAllocated) CmfError("Error: GetBuffer called before buffersize \"" + bufferSizeAttrName + "\" is set.");
                *buf = dataBuffer;
                *outputSize = allocatedSize;
            }
        private:
            /// @brief Name of the attribute corresponding to the buffer size
            std::string bufferSizeAttrName;
            
            /// @brief Name of the attribute corresponding to the buffer stride
            std::string strideName;
            
            /// @brief The data buffer
            char* dataBuffer;
            
            /// @brief The size of the data buffer
            size_t allocatedSize;
            
            /// @brief Set to true when the buffer is allocated, and false when it is freed
            bool bufferIsAllocated;
            
            /// @brief The symbol at the beginning of a symbolic invocation
            std::string invocationStart;
            
            /// @brief The symbol at the end of a symbolic invocation
            std::string invocationEnd;
            
            /// @brief The stride of the current buffer
            int stride;
            
            /// @brief The header format string
            std::string format;
            
            /// @brief The type of the data buffer
            VtkAttributableType::VtkAttributableType bufferType;
            
            /// @brief The name of the current object
            std::string className;
        friend class VtkBuffer;
    };
}

#endif
