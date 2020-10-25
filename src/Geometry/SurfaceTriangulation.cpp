#include "SurfaceTriangulation.h"
#include "StringUtils.h"
#include "DebugTools.hx"
#include "CmfError.h"
namespace cmf
{
    SurfaceTriangulation::SurfaceTriangulation(void)
    {
        geometryType = "SurfaceTriangulation";
        isDefined = false;
        requireFree = false;
    }
    
    void SurfaceTriangulation::DefineTriangulation(std::string filename)
    {
        TriangulationType::TriangulationType type = GetTriangulationTypeFromFilename(filename);
        switch(type)
        {
            case TriangulationType::Stl:
            {
                StlConverter converter(filename);
                PopulateGeometry(&converter);
                break;
            }
            default:
            {
                CmfError("Filename \"" + filename + "\" is of unsupported type.");
                break;
            }
        }
        isDefined = true;
        requireFree = true;
    }
    
    void SurfaceTriangulation::PopulateGeometry(ICmfGeometryConverter* converter)
    {
        converter->ConvertGeometry(this);
    }
    
    TriangulationType::TriangulationType SurfaceTriangulation::GetTriangulationTypeFromFilename(std::string filename)
    {
        std::string fileExtension = GetFileExtension(filename);
        if (fileExtension=="stl") return TriangulationType::Stl;
        return TriangulationType::Unsupported;
    }
    
    SurfaceTriangulation::~SurfaceTriangulation(void)
    {
        
    }
}