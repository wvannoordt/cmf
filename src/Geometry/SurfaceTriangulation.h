#ifndef CMF_SURFACE_TRIANGULATION_H
#define CMF_SURFACE_TRIANGULATION_H
#include "cmf.h"
#include "ICmfGeometryConverter.h"
#include "StlConverter.h"
#include "ConditionSet.h"
namespace cmf
{
    namespace TriangulationType
    {
        enum TriangulationType
        {
            Stl,
            Vtk,
            Unsupported
        };
    }

    inline static std::string TriangulationTypeStr(int id)
    {
        switch (id)
        {
            case TriangulationType::Stl: return "Stl";
            case TriangulationType::Vtk: return "Vtk";
            case TriangulationType::Unsupported: return "Unsupported";
        }
        return PTL_AUTO_ENUM_TERMINATOR;
    }
    class ICmfGeometryConverter;
    /// @brief A class representing a finite discretization of a surface
    /// @author WVN
    class SurfaceTriangulation : ICmfGeometry
    {
            friend class ICmfGeometryConverter;
            friend class StlConverter;
        public:
            /// @brief Constructor for SurfaceTriangulation class
            /// @author WVN
            SurfaceTriangulation(void);
            
            /// @brief Populates the triangulation data
            /// @param filename A surface triangulation file to be read
            /// @author WVN
            void DefineTriangulation(std::string filename);
            
            /// @brief Destructor for SurfaceTriangulation class
            /// @author WVN
            ~SurfaceTriangulation(void);
            
            /// @brief Determines whether the provided box intersects the boundary of the geometry
            /// @param bounds The bounds of the box to check (xmin, xmax, ymin, ymax, [zmin, zmax])
            /// @author WVN
            bool BoxIntersectsBoundary(double* bounds);
            
            /// @brief Writes the surface triangulation to a file
            /// @param filename The file to write
            /// @author WVN
            void WriteTriangulation(std::string filename);
        
        private:
            
            /// @brief Outputs the bounding boxes in the lookup table as a point cloud
            /// @param filename The file name to write to
            /// @author WVN            
            void OutputBoundingBoxesAsCloud(std::string filename);
            
            /// @brief Populates the triangulation data using the specified converter
            /// @param converter The provided ICmfGeometryConverter object
            /// @author WVN
            void PopulateGeometry(ICmfGeometryConverter* converter);
            
            /// @brief Gets a TriangulationType from a filename
            /// @param filename The filename to get extension of
            /// @author WVN
            TriangulationType::TriangulationType GetTriangulationTypeFromFilename(std::string filename);
            
            /// @brief Allocates the point buffer
            /// @param s Size to allocate
            /// @author WVN
            double* AllocatePointBuffer(size_t s);
            
            /// @brief Allocates the normal buffer
            /// @param s Size to allocate
            /// @author WVN
            double* AllocateNormalBuffer(size_t s);
            
            /// @brief Sets the number of faces
            /// @param n Value to set
            /// @author WVN
            void SetNumFaces(size_t n);
            
            /// @brief Computes the boundinb box for the triangulation
            /// @author WVN
            void ComputeBoundingBox(void);
            
            /// @brief Computes the lookup table for the current geometry
            /// @author WVN
            void ComputeLookupTable(void);
            
            /// @brief Gets the bounding box at the specified index in the lookup table
            /// @param idx Index input
            /// @param coordsBox Bounding box output
            /// @author WVN
            void GetBoxAtIndex(int* idx, double* coordsBox);
            
            /// @brief Gets the index range (inclusive) from a bounding box
            /// @param ijkLow Index output, lower, inclusive
            /// @param ijkHigh Index output, upper, inclusive
            /// @param boundingBox Bounding box input
            /// @author WVN
            void GetIndicesFromBoundingBox(double* boundingBox, int* ijkLow, int* ijkHigh);
            
            /// @brief dimensions of the lookup table
            int lookupTableDim[3];
            
            /// @brief bounding box of the triangulation
            double boundingBox[6];
            
            /// @brief Stores the startind index of the triangles in a given cell
            int* lookupTableBinStart;
            
            /// @brief Stores the number of triangles in the corresponding cell in the lookup table
            int* lookupTableBinCounts;
            
            /// @brief Stores the data for the lookup table
            int* lookupTableData;
            
            /// @brief Determines if DefineTriangulation has been called
            bool isDefined;
            
            /// @brief Determines if buffers need to be released on destruction
            bool requireFree;
            
            /// @brief The point buffer
            double* points;
            
            /// @brief The number of faces
            size_t numFaces;
            
            /// @brief The normal buffer
            double* normals;
            
            /// @brief A condition set that defined when a full definition has been achieved
            ConditionSet conditions;
    };
}

#endif