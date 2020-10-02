#ifndef VTK_BUFFER_H
#define VTK_BUFFER_H
#include "VtkFile.h"
namespace cmf
{
    class VtkBuffer
    {
        public:
            VtkBuffer(VtkAttributable* input);
            ~VtkBuffer(void);
            void operator << (double x);
            void operator << (int x);
            int Next(void);
        private:
            VtkAttributable* component;
            char* allocatedBuffer;
            size_t bufferSize;
            size_t curentOffset;
            int idx;
    };
}

#endif
