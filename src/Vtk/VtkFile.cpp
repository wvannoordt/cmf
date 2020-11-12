//https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
#include "VtkFile.h"
#include "StringUtils.h"
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
        scalarDimension = 0;
        hasScalarData = false;
    }
    
    VtkFile::VtkFile(void)
    {
        format = NULL;
        header = NULL;
        topology = NULL;
        version = NULL;
        filename = "NOFILE";
        isEmpty = true;
        scalarDimension = 0;
        hasScalarData = false;
    }

    void VtkFile::Write(void)
    {
        if (isEmpty)
        {
            CmfError("Attempted to output an empty VTK file object.");
        }
        std::ofstream myfile;
        topology->CheckHasAllAttributes();
        for (size_t i = 0; i < scalars.size(); i++)
        {
            scalars[i]->CheckHasAllAttributes();
        }
        myfile.open(filename.c_str());
        version->WriteToFile(myfile);
        header->WriteToFile(myfile);
        format->WriteToFile(myfile);
        topology->WriteToFile(myfile);
        if (hasScalarData) WriteScalarHeader(myfile);
        for (size_t i = 0; i < scalars.size(); i++)
        {
            scalars[i]->WriteToFile(myfile);
        }
        myfile.close();
    }
    
    void VtkFile::WriteScalarHeader(std::ofstream& myfile)
    {
        myfile << "CELL_DATA " << scalarDimension << std::endl;
    }

    VtkScalar* VtkFile::CreateScalar(std::string name, size_t numPoints)
    {
        if (!hasScalarData) scalarDimension = numPoints;
        else if (scalarDimension != numPoints) CmfError("Attempted to define a scalar \"" + name + "\" with dimension " + NiceCommaString(numPoints) + ". Current Vtk object has scalar dimension " + NiceCommaString(scalarDimension) + ".");
        VtkScalar* newScalar = new VtkScalar();
        scalars.push_back(newScalar);
        newScalar->Component("SCALARS")->SetAttribute("name", name);
        hasScalarData = true;
        return newScalar;
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
            for (int i = 0; i < scalars.size(); i++)
            {
                delete scalars[i];
            }
        }
    }
}
