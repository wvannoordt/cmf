#include "VtkVersion.h"
namespace cmf
{
    VtkVersion::VtkVersion(int major_in, int minor_in)
    {
        elementType = "version";
        major = major_in;
        minor = minor_in;
    }

    VtkVersion::VtkVersion(void)
    {
        elementType = "version";
        major = 0;
        minor = 0;
    }

    VtkVersion::~VtkVersion(void)
    {

    }

    void VtkVersion::WriteToFile(std::ofstream & myfile)
    {
        myfile << "# vtk DataFile Version " << major << "." << minor << std::endl;
    }
    
    void VtkVersion::ReadFromFile(std::ofstream & myfile)
    {

    }
}
