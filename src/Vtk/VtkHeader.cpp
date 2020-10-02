#include "VtkHeader.h"
namespace cmf
{
    VtkHeader::VtkHeader(std::string headerValue_in)
    {
        elementType = "header";
        headerValue = headerValue_in;
    }

    VtkHeader::VtkHeader(void)
    {
        elementType = "header";
        headerValue = "NOHEADERVALUE";
    }

    VtkHeader::~VtkHeader(void)
    {

    }

    void VtkHeader::WriteToFile(std::ofstream & myfile)
    {
        myfile << headerValue << std::endl;
    }

    void VtkHeader::ReadFromFile(std::ofstream & myfile)
    {
        
    }
}
