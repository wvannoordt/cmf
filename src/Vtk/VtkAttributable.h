#ifndef VTK_ATTR_H
#define VTK_ATTR_H
#define __VTKlongline "---------------------------------------------------------"
#define __VTKERROR(mystuff) {std::cout << __VTKlongline <<std::endl << "[VTK] Terminate called from " << __FILE__ <<  ", line (" << __LINE__ << "): " << mystuff << std::endl << __VTKlongline << std::endl; abort();}
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <map>

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
    };
    class VtkAttributable
    {
        public:
            bool RequiredAttributeExists(std::string name)
            {
                std::map<std::string,VtkAttributableType::VtkAttributableType>::iterator it = requiredAttributes.find(name);
                if(it != requiredAttributes.end())
                {
                   return true;
                }
                return false;
            }
            
            void AddRequiredAttribute(std::string name, VtkAttributableType::VtkAttributableType attribType)
            {
                if (RequiredAttributeExists(name)) __VTKERROR("Multiple definition of attribute " << name);
                requiredAttributes.insert({name, attribType});
            }
            
            VtkAttributable(void)
            {
                bufferIsAllocated = false;
                nextPointer = 0;
                AddRequiredAttribute("bufferSize", VtkAttributableType::longType);
            }
            
            ~VtkAttributable(void)
            {
                if (bufferIsAllocated) free(dataBuffer);
            }
            
        protected:
            char attributeBuffer[ANA_VTK_BUF_SIZE];
            int nextPointer;
            void* dataBuffer;
            std::map<std::string, VtkAttributableType::VtkAttributableType> requiredAttributes;
            std::map<std::string, VtkAttribute> attributes;
        private:
            bool bufferIsAllocated;
    };
}

#endif