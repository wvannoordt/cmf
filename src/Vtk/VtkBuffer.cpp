#include "VtkBuffer.h"
namespace Anaptric
{
    VtkBuffer::VtkBuffer(VtkAttributable* input)
    {
        component = input;
        component->GetBuffer(&allocatedBuffer, &bufferSize);
        curentOffset = 0;
        idx = 0;
    }

    VtkBuffer::~VtkBuffer(void){}

    void VtkBuffer::operator<< (double x)
    {
        idx++;
        *((double*)(allocatedBuffer+curentOffset)) = x;
        curentOffset+=(sizeof(double)/sizeof(char));
    }

    void VtkBuffer::operator<< (int x)
    {
        idx++;
        *((int*)(allocatedBuffer+curentOffset)) = x;
        curentOffset+=(sizeof(int)/sizeof(char));
    }

    int VtkBuffer::Next(void)
    {
        return idx;
    }
}
