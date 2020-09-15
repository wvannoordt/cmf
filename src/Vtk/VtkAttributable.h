#ifndef VTK_ATTR_H
#define VTK_ATTR_H
#include <iostream>
#define __VTKlongline "---------------------------------------------------------"
#define __VTKERROR(mystuff) {std::cout << __VTKlongline <<std::endl << "[VTK] Terminate called from " << __FILE__ <<  ", line (" << __LINE__ << "): " << mystuff << std::endl << __VTKlongline << std::endl; abort();}
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <map>
#include <cstring>

#define ANA_VTK_BUF_SIZE 4096

namespace Anaptric
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
        char* attrLocation;
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
                    if (!AttributeExists(it->first)) __VTKERROR("Cannot find required attribute \"" << it->first << "\" for object \"" << className << "\".");
                }
            }

            void AddRequiredAttribute(std::string name, VtkAttributableType::VtkAttributableType attribType)
            {
                if (RequiredAttributeExists(name)) __VTKERROR("Multiple definition of attribute " << name);
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
                newAttrValues->attrLocation = attributeBuffer+nextPointer;
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
                    case VtkAttributableType::stringType: __VTKERROR("Error: bufferType is set to incompatible type (" << AttrTypeStr(bufferType) << ").");
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
                    __VTKERROR("Stopping");
                }
                if ((requiredAttributes[name] != setType) && (!TypesAreCompatible(requiredAttributes[name], setType)))
                {
                    std::cout << "Error: Attempt to set attribute \"" << name << "\" to type " << std::endl;
                    std::cout << "\"" << AttrTypeStr(setType) << "\" but it is of type \"" << AttrTypeStr(requiredAttributes[name]) << "\"" << std::endl;
                    __VTKERROR("Stopping");
                }
                VtkAttribute newAttrValues;
                CreateNewAttribute(&newAttrValues, name, data, setType);
                attributes.insert({name, newAttrValues});
                if (name == bufferSizeAttrName)
                {
                    if (!TypesAreCompatible(setType, VtkAttributableType::longType))
                    {
                        __VTKERROR("Attempted to allocate dataBuffer using attribute \"" << name << "\" of improper type " << AttrTypeStr(setType) << ".");
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
                        __VTKERROR("Attempted to set stride using attribute \"" << name << "\" of improper type " << AttrTypeStr(setType) << ".");
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
            }

            ~VtkAttributable(void)
            {
                Destroy();
            }

            void Write(std::ofstream & myfile)
            {
                size_t elemSize = GetBufferElementSize();
                size_t totalEntries = allocatedSize/elemSize;
                std::cout << totalEntries << std::endl;
                switch (bufferType)
                {
                    case VtkAttributableType::intType:
                    {
                        for (size_t idx = 0; idx < totalEntries; idx++)
                        {
                            myfile << *((int*)(dataBuffer+idx*elemSize)) << " ";
                            if ((idx%stride==0)&&(idx>0)) myfile << std::endl;
                        }
                        break;
                    }
                    case VtkAttributableType::longType:
                    {
                        for (size_t idx = 0; idx < totalEntries; idx++)
                        {
                            myfile << *((size_t*)(dataBuffer+idx*elemSize)) << " ";
                            if ((idx%stride==0)&&(idx>0)) myfile << std::endl;
                        }
                        break;
                    }
                    case VtkAttributableType::doubleType:
                    {
                        for (size_t idx = 0; idx < totalEntries; idx++)
                        {
                            myfile << *((double*)(dataBuffer+idx*elemSize)) << " ";
                            if ((idx%stride==0)&&(idx>0)) myfile << std::endl;
                        }
                        break;
                    }
                }
            }

        protected:
            char attributeBuffer[ANA_VTK_BUF_SIZE];
            int nextPointer;
            std::map<std::string, VtkAttributableType::VtkAttributableType> requiredAttributes;
            std::map<std::string, VtkAttribute> attributes;
            void GetBuffer(char** buf, size_t* outputSize)
            {
                if (!bufferIsAllocated) __VTKERROR("Error: GetBuffer called before buffersize \"" << bufferSizeAttrName << "\" is set.");
                *buf = dataBuffer;
                *outputSize = allocatedSize;
            }
        private:
            std::string bufferSizeAttrName, strideName;
            char* dataBuffer;
            size_t allocatedSize;
            bool bufferIsAllocated;
            int stride;
            VtkAttributableType::VtkAttributableType bufferType;
            std::string className;
        friend class VtkBuffer;
    };
}

#endif
