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
        
        private:
            
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