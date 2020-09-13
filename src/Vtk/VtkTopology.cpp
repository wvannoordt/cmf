#include "VtkTopology.h"
namespace Anaptric
{
    VtkTopology::VtkTopology(VtkTopologyType::VtkTopologyType topologyType_in)
    {
        elementType = "topology";
        topologyType = topologyType_in;
        uninitialized = false;
        InitTopology();
    }

    void VtkTopology::InitTopology(void)
    {
        switch (topologyType)
        {
            case VtkTopologyType::structuredPoints:
            {
                __VTKERROR("structuredPoints not implemented");
                collection.AddAttributable("DATASET", VtkAttributableType::intType);
                VtkAttributable* dataset = collection.GetAttributable("DATASET");
                dataset->AddRequiredAttribute("numpnts_x", VtkAttributableType::intType);
                dataset->AddRequiredAttribute("numpnts_y", VtkAttributableType::intType);
                dataset->AddRequiredAttribute("numpnts_z", VtkAttributableType::intType);
                dataset->AddRequiredAttribute("origin_x",  VtkAttributableType::doubleType);
                dataset->AddRequiredAttribute("origin_y",  VtkAttributableType::doubleType);
                dataset->AddRequiredAttribute("origin_z",  VtkAttributableType::doubleType);
                dataset->AddRequiredAttribute("spacing_x", VtkAttributableType::doubleType);
                dataset->AddRequiredAttribute("spacing_y", VtkAttributableType::doubleType);
                dataset->AddRequiredAttribute("spacing_z", VtkAttributableType::doubleType);
            }
            case VtkTopologyType::structuredGrid:
            {
                __VTKERROR("structuredGrid not implemented");
                break;
            }
            case VtkTopologyType::unstructuredGrid:
            {
                VtkAttributable* dataset = collection.AddAttributable("DATASET", VtkAttributableType::doubleType);
                dataset->AddRequiredAttribute("numPoints", VtkAttributableType::longType);
                dataset->AddRequiredAttribute("bufferCount", VtkAttributableType::longType);
                VtkAttributable* cells = collection.AddAttributable("CELLS", VtkAttributableType::intType);
                cells->AddRequiredAttribute("numPoints", VtkAttributableType::longType);
                cells->AddRequiredAttribute("totalEntries", VtkAttributableType::longType);
                cells->AddRequiredAttribute("bufferCount", VtkAttributableType::longType);
                VtkAttributable* cellTypes = collection.AddAttributable("CELL_TYPES", VtkAttributableType::intType);
                cellTypes->AddRequiredAttribute("numPoints", VtkAttributableType::longType);
                cellTypes->AddRequiredAttribute("bufferCount", VtkAttributableType::longType);
                break;
            }
            case VtkTopologyType::polydata:
            {
                __VTKERROR("polydata not implemented");
                break;
            }
            case VtkTopologyType::rectilinearGrid:
            {
                __VTKERROR("rectilinearGrid not implemented");
                break;
            }
            case VtkTopologyType::field:
            {
                __VTKERROR("field not implemented");
                break;
            }
        }
    }

    VtkTopology::VtkTopology(void)
    {
        elementType = "topology";
        VtkTopologyType::unstructuredGrid;
        uninitialized = true;
    }

    VtkTopology::~VtkTopology(void)
    {

    }

    void VtkTopology::WriteToFile(std::ofstream & myfile)
    {
        collection.WriteAll(myfile);
    }

    void VtkTopology::ReadFromFile(std::ofstream & myfile)
    {

    }
}
