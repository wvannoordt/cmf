#include "SurfaceTriangulation.h"
#include "StringUtils.h"
#include "DebugTools.hx"
#include "CmfError.h"
#include "CmfGC.h"
#include "Utils.hx"
#include <cmath>
#include "Vec.h"
namespace cmf
{
    SurfaceTriangulation::SurfaceTriangulation(void)
    {
        geometryType = "SurfaceTriangulation";
        isDefined = false;
        requireFree = false;
        numFaces = 0;
        __dloop(lookupTableDim[d]=0);
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
                StlConverter converter(filename, this);
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
    
    void SurfaceTriangulation::WriteTriangulation(std::string filename)
    {
        if (!isDefined) CmfError("Attempted to output surface triangulation \"" + filename + "\", but surface has not been defined! (see SurfaceTriangulation::DefineTriangulation)");
        TriangulationType::TriangulationType type = GetTriangulationTypeFromFilename(filename);
        switch(type)
        {
            case TriangulationType::Stl:
            {
                StlConverter converter(filename, this);
                converter.SaveGeometry();
                break;
            }
            default:
            {
                CmfError("Filename \"" + filename + "\" is of unsupported type.");
                break;
            }
        }
    }
    
    void SurfaceTriangulation::PopulateGeometry(ICmfGeometryConverter* converter)
    {
        converter->ConvertGeometry();
        if (!conditions.CheckAll())
        {
            std::string message = "Implementation of a derived class of ICmfGeometryConverter is underdefined!";
            message = message + "\nThe following conditions were not met:\n";
            conditions.WriteUnmetConditions(message);
            CmfError(message);
        }
        ComputeLookupTable();
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
        points = (double*)Cmf_Alloc(s);
        return points;
    }
    
    double* SurfaceTriangulation::AllocateNormalBuffer(size_t s)
    {
        conditions.SetCondition("AllocateNormalBuffer", true);
        normals = (double*)Cmf_Alloc(s);
        return normals;
    }
    
    void SurfaceTriangulation::SetNumFaces(size_t n)
    {
        numFaces = n;
        conditions.SetCondition("SetNumFaces", true);
    }
    void SurfaceTriangulation::ComputeBoundingBox(void)
    {
        for (int d = 0; d < 3; d++){boundingBox[2*d] = 1e50;}
        for (int d = 0; d < 3; d++){boundingBox[2*d+1] = -1e50;}
        double* p1;
        double* p2;
        double* p3;
        for (size_t face = 0; face < numFaces; face++)
        {
            p1 = points + 9*face;
            p2 = points + 9*face+3;
            p3 = points + 9*face+6;
            for (int d = 0; d < 3; d++){boundingBox[2*d] = CMFMIN(boundingBox[2*d],p1[d]);}
            for (int d = 0; d < 3; d++){boundingBox[2*d] = CMFMIN(boundingBox[2*d],p2[d]);}
            for (int d = 0; d < 3; d++){boundingBox[2*d] = CMFMIN(boundingBox[2*d],p3[d]);}
            for (int d = 0; d < 3; d++){boundingBox[2*d+1] = CMFMAX(boundingBox[2*d+1],p1[d]);}
            for (int d = 0; d < 3; d++){boundingBox[2*d+1] = CMFMAX(boundingBox[2*d+1],p2[d]);}
            for (int d = 0; d < 3; d++){boundingBox[2*d+1] = CMFMAX(boundingBox[2*d+1],p3[d]);}
        }
        WriteLine(3, "xmin = " + std::to_string(boundingBox[0]));
        WriteLine(3, "xmax = " + std::to_string(boundingBox[1]));
        WriteLine(3, "ymin = " + std::to_string(boundingBox[2]));
        WriteLine(3, "ymax = " + std::to_string(boundingBox[3]));
        WriteLine(3, "zmin = " + std::to_string(boundingBox[4]));
        WriteLine(3, "zmax = " + std::to_string(boundingBox[5]));
    }
    
    void SurfaceTriangulation::ComputeLookupTable(void)
    {
        //temporary
        WriteLine(0, "WARNING: HARDCODED VALUE FOR LOOKUP TABLE DIMENSIONS!");
        __dloop(lookupTableDim[d]=55);
        size_t totalNumCells = 1;
        __dloop(totalNumCells*=lookupTableDim[d]);
        lookupTableBinCounts = (int*)Cmf_Alloc(totalNumCells*sizeof(int));
        lookupTableBinStart  = (int*)Cmf_Alloc(totalNumCells*sizeof(int));
        int* lookupTableBinIdx  = (int*)Cmf_Alloc(totalNumCells*sizeof(int));
        ComputeBoundingBox();
        size_t totalLookupTableSize = 0;
        double* p1;
        double* p2;
        double* p3;
        double* normalVec;
        double faceBoundingBox[6];
        int ijkLow[3];
        int ijkHigh[3];
        double gridBoxBounds[6];
        int idx[3];
        int index;
        WriteLine(4, "Lookup table start");
        for (int countMode = 0; countMode<2; countMode++)
        {
            if (countMode==0)
            {
                for (size_t cellIdxTemp = 0; cellIdxTemp < totalNumCells; cellIdxTemp++)
                {
                    lookupTableBinCounts[cellIdxTemp]=0;
                    lookupTableBinIdx[cellIdxTemp]=0;
                }
            }
            for (size_t face = 0; face < numFaces; face++)
            {
                int numFoundForThisFace = 0;
                p1 = points + 9*face;
                p2 = points + 9*face+3;
                p3 = points + 9*face+6;
                normalVec  = normals + 3*face;
                
                for (int d = 0; d < 3; d++)
                {
                    ijkLow[d] = 0;
                    ijkHigh[d] = 0;
                    faceBoundingBox[2*d] = CMFMIN(CMFMIN(p1[d], p2[d]), p3[d]);
                    faceBoundingBox[2*d+1] = CMFMAX(CMFMAX(p1[d], p2[d]), p3[d]);
                }
                GetIndicesFromBoundingBox(faceBoundingBox, ijkLow, ijkHigh);
                for (int i = ijkLow[0]; i <= ijkHigh[0]; i++)
                {
                    for (int j = ijkLow[1]; j <= ijkHigh[1]; j++)
                    {
                        for (int k = ijkLow[2]; k <= ijkHigh[2]; k++)
                        {
                            int ijk[3]={i,j,k};
                            index = Idx2Dim3(lookupTableDim,ijk);
                            GetBoxAtIndex(ijk, gridBoxBounds);
                            if (TriangleIntersectsBox(p1, p2, p3, normalVec, gridBoxBounds))
                            {
                                numFoundForThisFace++;
                                if (countMode==0)
                                {
                                    totalLookupTableSize++;
                                    lookupTableBinCounts[index]++;
                                }
                                else
                                {
                                    lookupTableData[lookupTableBinStart[index]+lookupTableBinIdx[index]] = (int)face;
                                    lookupTableBinIdx[index]++;
                                }
                            }
                        }
                    }
                }
                if (numFoundForThisFace==0) CmfError("Error: found a face (" + std::to_string(face) + ") with no bounding boxes...");
            }
            if (countMode==0)
            {
                lookupTableData = (int*)Cmf_Alloc(totalLookupTableSize*sizeof(int));
                for (int cellIdx = 0; cellIdx < totalNumCells; cellIdx++) lookupTableBinStart[cellIdx] = 0;
                for (int cellIdx = 1; cellIdx < totalNumCells; cellIdx++)
                {
                    lookupTableBinStart[cellIdx] = lookupTableBinStart[cellIdx-1]+lookupTableBinCounts[cellIdx-1];
                }
            }
        }
        Cmf_Free(lookupTableBinIdx);
    }
    
    void SurfaceTriangulation::OutputBoundingBoxesAsCloud(std::string filename)
    {
        size_t totalNumCells=1;
        __dloop(totalNumCells*=lookupTableDim[d]);
        std::ofstream myfile;
		myfile.open(filename);
        for (int i = 0; i < totalNumCells; i++)
        {
            if (lookupTableBinCounts[i]>0)
            {
                double coordsBox[6];
                double cornereeee[3];
                int idxwww[3];
                Dim2Idx3(i, lookupTableDim, idxwww);
                GetBoxAtIndex(idxwww, coordsBox);
                for (int y = 0; y < 8; y++)
                {
                    cornereeee[0] = coordsBox[0 + ((0>>0)&1)];
                    cornereeee[1] = coordsBox[2 + ((0>>1)&1)];
                    cornereeee[2] = coordsBox[4 + ((0>>2)&1)];
                    myfile << cornereeee[0] << ", " << cornereeee[1] << ", " << cornereeee[2] << std::endl;
                }
            }
        }
        myfile.close();
    }
    
    bool SurfaceTriangulation::BoxIntersectsBoundary(double* bounds)
    {
        int ijkLow[3];
        int ijkHigh[3];
        double* p1;
        double* p2;
        double* p3;
        double* normalVec;
        size_t index;
        GetIndicesFromBoundingBox(bounds, ijkLow, ijkHigh);
        for (int i = ijkLow[0]; i <= ijkHigh[0]; i++)
        {
            for (int j = ijkLow[1]; j <= ijkHigh[1]; j++)
            {
                for (int k = ijkLow[2]; k <= ijkHigh[2]; k++)
                {
                    int ijk[3]={i,j,k};
                    if (CheckBounds3(ijk, lookupTableDim))
                    {
                        index = Idx2Dim3(lookupTableDim,ijk);
                        int totalNumFaces = lookupTableBinCounts[index];
                        for (int faceNum = 0; faceNum < totalNumFaces; faceNum++)
                        {
                            int face = lookupTableData[lookupTableBinStart[index]+faceNum];
                            p1 = points + 9*face;
                            p2 = points + 9*face+3;
                            p3 = points + 9*face+6;
                            normalVec  = normals + 3*face;
                            if (TriangleIntersectsBox(p1, p2, p3, normalVec, bounds)) return true;
                        }
                    }
                }
            }
        }
        return false;
    }
    
    void SurfaceTriangulation::GetIndicesFromBoundingBox(double* boundingBoxIn, int* ijkLow, int* ijkHigh)
    {
        for (int d = 0; d < 3; d++)
        {
            ijkLow[d]  = floor(lookupTableDim[d]*(boundingBoxIn[2*d]-boundingBox[2*d])/(boundingBox[2*d+1]-boundingBox[2*d]));
            int l = floor(lookupTableDim[d]*(boundingBoxIn[2*d+1]-boundingBox[2*d])/(boundingBox[2*d+1]-boundingBox[2*d]));
            ijkHigh[d] = CMFMIN(l,(lookupTableDim[d]-1));
            ijkLow[d] = CMFMIN(ijkLow[d],(lookupTableDim[d]-1));
        }
    }
    
    void SurfaceTriangulation::GetBoxAtIndex(int* idx, double* coordsBox)
    {
        const double scaleBoxFactor = 1.001;
        double delta[3];
        double avg[3];
        coordsBox[0] = boundingBox[0] + (double)idx[0]*(boundingBox[1]-boundingBox[0])/lookupTableDim[0];
        coordsBox[1] = boundingBox[0] + (double)(idx[0]+1)*(boundingBox[1]-boundingBox[0])/lookupTableDim[0];
        coordsBox[2] = boundingBox[2] + (double)idx[1]*(boundingBox[3]-boundingBox[2])/lookupTableDim[1];
        coordsBox[3] = boundingBox[2] + (double)(idx[1]+1)*(boundingBox[3]-boundingBox[2])/lookupTableDim[1];
        coordsBox[4] = boundingBox[4] + (double)idx[2]*(boundingBox[5]-boundingBox[4])/lookupTableDim[2];
        coordsBox[5] = boundingBox[4] + (double)(idx[2]+1)*(boundingBox[5]-boundingBox[4])/lookupTableDim[2];
        delta[0] = (coordsBox[1] - coordsBox[0])*0.5;
        delta[1] = (coordsBox[3] - coordsBox[2])*0.5;
        delta[2] = (coordsBox[5] - coordsBox[4])*0.5;
        avg[0] = (coordsBox[1]+coordsBox[0])*0.5;
        avg[1] = (coordsBox[3]+coordsBox[2])*0.5;
        avg[2] = (coordsBox[5]+coordsBox[4])*0.5;
        coordsBox[0] = avg[0] - scaleBoxFactor*delta[0];
        coordsBox[1] = avg[0] + scaleBoxFactor*delta[0];
        coordsBox[2] = avg[1] - scaleBoxFactor*delta[1];
        coordsBox[3] = avg[1] + scaleBoxFactor*delta[1];
        coordsBox[4] = avg[2] - scaleBoxFactor*delta[2];
        coordsBox[5] = avg[2] + scaleBoxFactor*delta[2];
    }
    
    SurfaceTriangulation::~SurfaceTriangulation(void)
    {
        if (requireFree)
        {
            Cmf_Free(normals);
            Cmf_Free(points);
            Cmf_Free(lookupTableBinCounts);
            Cmf_Free(lookupTableData);
            Cmf_Free(lookupTableBinStart);
        }
    }
}