#include "VtkFile.h"

namespace Anaptric
{
    VtkFile::VtkFile(std::string filename_in, VtkFormatType::VtkFormatType formType, VtkTopologyType::VtkTopologyType topType)
    {
        format = VtkFileFormat(formType);
        header = VtkHeader("Anaptric");
        topology = VtkTopology(topType);
        version = VtkVersion(3, 0);
        filename = filename_in;
    }

    void VtkFile::Write(void)
    {
        std::ofstream myfile;
        myfile.open(filename.c_str());
        version.WriteToFile(myfile);
        header.WriteToFile(myfile);
        format.WriteToFile(myfile);
        topology.WriteToFile(myfile);
        for (size_t i = 0; i < data.size(); i++)
        {
            data[i]->WriteToFile(myfile);
        }
        myfile.close();
    }

    void VtkFile::SetAdjacency(size_t numEdges, size_t numEntries)
    {
        topology.SetAdjacency(numEdges, numEntries);
    }

    void VtkFile::AddPoint(double x, double y)
    {
        topology.AddPoint(x, y);
    }

    void VtkFile::AddPoint(double x, double y, double z)
    {
        topology.AddPoint(x, y, z);
    }

    void VtkFile::SetPointCount(size_t pointNum)
    {
        topology.SetPointCount(pointNum);
    }

    void VtkFile::SetPointCount(size_t pointNumX, size_t pointNumY, size_t pointNumZ)
    {
        topology.SetPointCount(pointNumX, pointNumY, pointNumZ);
    }

    VtkFile::~VtkFile(void)
    {

    }
}
