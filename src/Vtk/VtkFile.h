#ifndef VTK_FILE_H
#define VTK_FILE_H

#include <string>
#include <vector>
#include "VtkFileFormat.h"
#include "VtkData.h"
#include "VtkHeader.h"
#include "VtkTopology.h"
#include "VtkVersion.h"
#include "VtkBuffer.h"
#include "CmfError.h"

namespace cmf
{
    namespace VtkCellType
    {
        enum VtkCellType
        {
            none,
            vertex,
            polyVertex,
            line,
            polyLine,
            triangle,
            triangleStrip,
            polygon,
            pixel,
            quad,
            tetra,
            voxel,
            hexahedron,
            wedge,
            pyramid,
            quadraticEdge,
            quadraticTriangle,
            quadraticQuad,
            quadraticTetra,
            quadraticHexahedron
        };
    }
    class VtkFile
    {
        public:
            VtkFile(std::string filename_in, VtkFormatType::VtkFormatType formType, VtkTopologyType::VtkTopologyType topType);
            VtkFile(void);
            ~VtkFile(void);
            void Write(void);
            VtkTopology* & Mesh(void);
        private:
            std::string filename;
            VtkVersion* version;
            VtkFileFormat* format;
            VtkHeader* header;
            VtkTopology* topology;
            std::vector<VtkData*> data;
            bool isEmpty;
    };
}

#endif
