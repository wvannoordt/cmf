//https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
#include "VtkFile.h"
namespace cmf
{
    VtkFile::VtkFile(std::string filename_in, VtkFormatType::VtkFormatType formType, VtkTopologyType::VtkTopologyType topType)
    {
        format = new VtkFileFormat(formType);
        header = new VtkHeader("cmf");
        topology = new VtkTopology(topType);
        version = new VtkVersion(3, 0);
        filename = filename_in;
        isEmpty = false;
    }
    
    VtkFile::VtkFile(void)
    {
        format = NULL;
        header = NULL;
        topology = NULL;
        version = NULL;
        filename = "NOFILE";
        isEmpty = true;
    }

    void VtkFile::Write(void)
    {
        if (isEmpty)
        {
            CmfError("Attempted to output an empty VTK file object.");
        }
        std::ofstream myfile;
        topology->CheckHasAllAttributes();
        for (size_t i = 0; i < data.size(); i++)
        {
            data[i]->CheckHasAllAttributes();
        }
        myfile.open(filename.c_str());
        version->WriteToFile(myfile);
        header->WriteToFile(myfile);
        format->WriteToFile(myfile);
        topology->WriteToFile(myfile);
        for (size_t i = 0; i < data.size(); i++)
        {
            data[i]->WriteToFile(myfile);
        }
        myfile.close();
    }

    VtkTopology* & VtkFile::Mesh(void)
    {
        return topology;
    }

    VtkFile::~VtkFile(void)
    {
        if (!isEmpty)
        {
            delete format;
            delete header;
            delete topology;
            delete version;
        }
    }
}
