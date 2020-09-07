#include "VtkFileFormat.h"
namespace Anaptric
{
    VtkFileFormat::VtkFileFormat(VtkFormatType::VtkFormatType formType_in)
    {
        elementType = "fileformat";
        formType = formType_in;
    }
    
    VtkFileFormat::VtkFileFormat(void)
    {
        elementType = "fileformat";
        formType = VtkFormatType::ascii;
    }
    
    VtkFileFormat::~VtkFileFormat(void)
    {
        
    }
    
    void VtkFileFormat::WriteToFile(std::ofstream & myfile)
    {
        switch (formType)
        {
            case VtkFormatType::ascii:
            {
                myfile << "ASCII" << std::endl;
                break;
            }
            case VtkFormatType::binary:
            {
                myfile << "BINARY" << std::endl;
                break;
            }
        }
    }
    
    void VtkFileFormat::ReadFromFile(std::ofstream & myfile)
    {
        
    }
}