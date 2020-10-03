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

    struct VtkAttribute
    {
        std::string name;
        VtkAttributableType::VtkAttributableType attrType;
        size_t attrOffset;
        size_t varSize;
    };

    class VtkAttributable
    {
        public:
            void BufferSizeIs(std::string bufferSizeName)
            {
                bufferSizeAttrName = bufferSizeName;
            }

            void StrideIs(std::string strideName_in)
            {
                strideName = strideName_in;
            }

            bool RequiredAttributeExists(std::string name)
            {
                std::map<std::string,VtkAttributableType::VtkAttributableType>::iterator it = requiredAttributes.find(name);
                if(it != requiredAttributes.end())
                {
                   return true;
                }
                return false;
            }

            bool AttributeExists(std::string name)
            {
                std::map<std::string,VtkAttribute>::iterator it = attributes.find(name);
                if(it != attributes.end())
                {
                   return true;
                }
                return false;
            }

            void EnforceAllRequiredAttributes(void)
            {
                for (std::map<std::string,VtkAttributableType::VtkAttributableType>::iterator it = requiredAttributes.begin(); it != requiredAttributes.end(); it++)
                {
                    if (!AttributeExists(it->first)) CmfError("Cannot find required attribute \"" + it->first + "\" for object \"" + className + "\".");
                }
            }

            void AddRequiredAttribute(std::string name, VtkAttributableType::VtkAttributableType attribType)
            {
                if (RequiredAttributeExists(name)) CmfError("Multiple definition of attribute " + name);
                requiredAttributes.insert({name, attribType});
            }

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

            bool TypesAreCompatible(VtkAttributableType::VtkAttributableType a, VtkAttributableType::VtkAttributableType b)
            {
                if ((a==VtkAttributableType::intType) && (b==VtkAttributableType::longType)) return true;
                if ((b==VtkAttributableType::intType) && (a==VtkAttributableType::longType)) return true;
                return false;
            }

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
                    dataBuffer = (char*)malloc(allocsize);
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

            void GetBuffer(int**    bufOut, size_t* outputSize) {*outputSize = allocatedSize; *bufOut = (int*)    dataBuffer;}
            void GetBuffer(int**    bufOut)                     {                             *bufOut = (int*)    dataBuffer;}
            void GetBuffer(double** bufOut, size_t* outputSize) {*outputSize = allocatedSize; *bufOut = (double*) dataBuffer;}
            void GetBuffer(double** bufOut)                     {                             *bufOut = (double*) dataBuffer;}

            void SetAttribute(std::string name, int data)         {SetAttribute(name, (void*)&data, VtkAttributableType::intType);   }
            void SetAttribute(std::string name, double data)      {SetAttribute(name, (void*)&data, VtkAttributableType::doubleType);}
            void SetAttribute(std::string name, size_t data)      {SetAttribute(name, (void*)&data, VtkAttributableType::longType);  }
            void SetAttribute(std::string name, std::string data) {SetAttribute(name, (void*)&data, VtkAttributableType::stringType);}

            void Destroy(void)
            {
                if (bufferIsAllocated)
                {
                    bufferIsAllocated = false;
                    free(dataBuffer);
                }
            }

            void SetFormat(std::string format_in)
            {
                format = format_in;
            }

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

            ~VtkAttributable(void)
            {
                Destroy();
            }

            bool PositionIsStart(size_t i, std::string str)
            {
                if (i < invocationStart.length()-1) return false;
                return (str.substr(i-invocationStart.length()+1, invocationStart.length())==invocationStart);
            }

            bool PositionIsEnd(size_t i, std::string str)
            {
                if (i+invocationEnd.length()-1 >= (str.length())) return false;
                return (str.substr(i, invocationEnd.length()) == invocationEnd);
            }

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

            std::string HeaderString(void) {return HeaderString(format, 0);}

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
            char attributeBuffer[CMF_VTK_BUF_SIZE] = {0};
            int nextPointer;
            std::map<std::string, VtkAttributableType::VtkAttributableType> requiredAttributes;
            std::map<std::string, VtkAttribute> attributes;
            void GetBuffer(char** buf, size_t* outputSize)
            {
                if (!bufferIsAllocated) CmfError("Error: GetBuffer called before buffersize \"" + bufferSizeAttrName + "\" is set.");
                *buf = dataBuffer;
                *outputSize = allocatedSize;
            }
        private:
            std::string bufferSizeAttrName, strideName;
            char* dataBuffer;
            size_t allocatedSize;
            bool bufferIsAllocated;
            std::string invocationStart, invocationEnd;
            int stride;
            std::string format;
            VtkAttributableType::VtkAttributableType bufferType;
            std::string className;
        friend class VtkBuffer;
    };
}

#endif
