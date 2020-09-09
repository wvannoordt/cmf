#ifndef VTK_TOPOLOGY_H
#define VTK_TOPOLOGY_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "VtkElement.h"
#include "VtkAttributable.h"

namespace Anaptric
{
    namespace VtkTopologyType
    {
        enum VtkTopologyType
        {
            structuredPoints,
            structuredGrid,
            unstructuredGrid,
            polydata,
            rectilinearGrid,
            field
        };
    }

    inline static std::string VtkTopologyTypeStr(VtkTopologyType::VtkTopologyType ttype)
    {
        switch (ttype)
        {
            case VtkTopologyType::structuredPoints: return "STRUCTURED_POINTS";
            case VtkTopologyType::structuredGrid:   return "STRUCTURED_GRID";
            case VtkTopologyType::unstructuredGrid: return "UNSTRUCTURED_GRID";
            case VtkTopologyType::polydata:         return "POLYDATA";
            case VtkTopologyType::rectilinearGrid:  return "RECTILINEAR_GRID";
            case VtkTopologyType::field:            return "FIELD";
        }
        return "NONE";
    }

    class VtkTopology : public VtkElement, public 
    {
        public:
            VtkTopology(VtkTopologyType::VtkTopologyType topologyType_in);
            VtkTopology(void);
            ~VtkTopology(void);
            void WriteToFile(std::ofstream & myfile);
            void ReadFromFile(std::ofstream & myfile);
            void SetPointCount(size_t pointNum);
            void SetPointCount(size_t pointNumX, size_t pointNumY, size_t pointNumZ);
            void AddPoint(double x, double y);
            void AddPoint(double x, double y, double z);
        private:
            VtkTopologyType::VtkTopologyType topologyType;
    };
}

#endif
