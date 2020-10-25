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
        numFaces = 0;
        conditions.AddCondition("SetNumFaces");
        conditions.AddCondition("AllocatePointBuffer");
        conditions.AddCondition("AllocateNormalBuffer");
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
        if (!conditions.CheckAll())
        {
            std::string message = "Implementation of a derived class of ICmfGeometryConverter is underdefined!";
            message = message + "\nThe following conditions were not met:\n";
            conditions.WriteUnmetConditions(message);
            CmfError(message);
        }
    }
    
    TriangulationType::TriangulationType SurfaceTriangulation::GetTriangulationTypeFromFilename(std::string filename)
    {
        std::string fileExtension = GetFileExtension(filename);
        if (fileExtension=="stl") return TriangulationType::Stl;
        return TriangulationType::Unsupported;
    }
    
    double* SurfaceTriangulation::AllocatePointBuffer(size_t s)
    {
        conditions.SetCondition("AllocatePointBuffer", true);
        points = (double*)malloc(s);
        return points;
    }
    
    double* SurfaceTriangulation::AllocateNormalBuffer(size_t s)
    {
        conditions.SetCondition("AllocateNormalBuffer", true);
        normals = (double*)malloc(s);
        return normals;
    }
    
    void SurfaceTriangulation::SetNumFaces(size_t n)
    {
        numFaces = n;
        conditions.SetCondition("SetNumFaces", true);
    }
    
    SurfaceTriangulation::~SurfaceTriangulation(void)
    {
        
    }
}