#ifndef VTK_FILE_H
#define VTK_FILE_H

#include <string>
#include <vector>
#include "VtkFileFormat.h"
#include "VtkData.h"
#include "VtkHeader.h"
#include "VtkTopology.h"
#include "VtkVersion.h"

namespace Anaptric
{
    class VtkFile
    {
        public:
            VtkFile(std::string filename_in, VtkFormatType::VtkFormatType formType, VtkTopologyType::VtkTopologyType topType);
            ~VtkFile(void);
            void Write(void);
            void SetPointCount(size_t pointNum);
            void SetPointCount(size_t pointNumX, size_t pointNumY, size_t pointNumZ);
            void AddPoint(double x, double y);
            void AddPoint(double x, double y, double z);
            void SetAdjacency(size_t numEdges, size_t numEntries);
        private:
            std::string filename;
            VtkVersion version;
            VtkFileFormat format;
            VtkHeader header;
            VtkTopology topology;
            std::vector<VtkData*> data;
    };
}

#endif
