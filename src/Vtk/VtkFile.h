#ifndef VTK_FILE_H
#define VTK_FILE_H

#include <string>
#include <vector>
#include "VtkFileFormat.h"
#include "VtkScalar.h"
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
    /// @brief Class representing a VTK file, used for VTK output
    /// @author WVN
    class VtkFile
    {
        public:
            /// @brief Constructor for VtkFile
            /// @param filename_in File name to be written to
            /// @param formType File type of the VTK file
            /// @param topType Topology (grid) type of the VTK file
            /// \see VtkFormatType VtkTopologyType
            /// @author WVN
            VtkFile(std::string filename_in, VtkFormatType::VtkFormatType formType, VtkTopologyType::VtkTopologyType topType);
            
            /// @brief Empty constructor for VtkFile
            /// @author WVN
            VtkFile(void);
            
            /// @brief Destructor for VtkFile
            /// @author WVN
            ~VtkFile(void);
            
            /// @brief Write the file to the filename provided in the constructor
            /// @author WVN
            void Write(void);
            
            /// @brief Returns the pointer to the underlying VtkTopology object.
            /// @author WVN
            VtkTopology* & Mesh(void);
            
            /// @brief Creates a VtkScalar with the given name
            /// @param name The name of the new scalar
            /// @param numPoints The number of points in the scalar field
            /// @author WVN
            VtkScalar* CreateScalar(std::string name, size_t numPoints);
            
        private:
            
            /// @brief Writes the scalar header to the output file
            /// @param myfile The file stream to write the header to
            /// @author WVN
            void WriteScalarHeader(std::ofstream& myfile);
            
            /// @brief File name of the file to be written to
            std::string filename;
            
            /// @brief The VTK version object
            VtkVersion* version;
            
            /// @brief The VTK format object
            VtkFileFormat* format;
            
            /// @brief The VTK header object
            VtkHeader* header;
            
            /// @brief The VTK topology object
            VtkTopology* topology;
            
            /// @brief Indicates if the Vtk file has scalar data in it
            bool hasScalarData;
            
            /// @brief The dimension of the scalar objects
            size_t scalarDimension;
            
            /// @brief A list of VTK data array objects
            std::vector<VtkScalar*> scalars;
            
            /// @brief Set to ture in empty constructor
            bool isEmpty;
    };
}

#endif
