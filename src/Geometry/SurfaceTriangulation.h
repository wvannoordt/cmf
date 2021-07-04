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
    /// @brief A class representing a finite discretization of a surface. This class is not intended to handle any data
    /// over a surface!
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
            
            /// @brief Determines whether the provided box intersects the boundary of the geometry
            /// @param bounds The bounds of the box to check (xmin, xmax, ymin, ymax, zmin, zmax)
            /// @author WVN
            bool BoxIntersectsBoundary(Vec<double, 6> bounds);
            
            /// @brief Writes the surface triangulation to a file
            /// @param filename The file to write
            /// @author WVN
            void WriteTriangulation(std::string filename);
            
            
            /// @brief Returns the number of faces
            /// @author WVN
            size_t GetNumFaces(void);
            
            /// @brief (Debugging only) Writes the triangles that intersect a given box as an STL file.
            /// Returns the subset of intersecting faces
            /// @param bounds The box to check (xmin xmax ymin ymax [zmin zmax])
            /// @param filename The file to write
            /// @author WVN
            std::vector<int> OutputTrianglesIntersectingBoxAsStl(double* bounds, std::string filename);
            
            /// @brief (Debugging only) Copies the points and normal vector of a triangle to the
            /// provided arrays (mut be of size 3 or greater)
            /// @param faceIndex The index of the face to fetch
            /// @param p1 The array to fill with the first point info
            /// @param p2 The array to fill with the second point info
            /// @param p3 The array to fill with the third point info
            /// @param n A pointer to the array to fill with the normal vector info
            /// @author WVN
            void CopyFaceInfo(int faceIndex, double* p1, double* p2, double* p3, double* n);
            
            /// @brief Accessor for vertex data
            /// @param i The face index
            /// @param v The vertex index (0-2)
            /// @param p The coordinate index (0-2)
            /// @author WVN
            double& operator () (size_t i, int v, int c)
            {
                return points[9*i + 3*v+c];
            }
        
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