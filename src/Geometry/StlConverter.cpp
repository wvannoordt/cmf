#include "StlConverter.h"
#include "CmfScreen.h"
#include "CmfError.h"
#include "StringUtils.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include "CmfGC.h"
#include "Utils.hx"
#include "DebugTools.hx"
namespace cmf
{
    StlConverter::StlConverter(std::string filename)
    {
        WriteLine(1, "Reading STL file \"" + filename + "\"...");
        int num = 1;
        bigEndian = !(*(char*)&num==1);
        if (bigEndian) CmfError("Unable to handle big-endian data for now. This should be a simple fix.");
        requireFree = false;
        facetCount = -999;
        xmin =  1e30;
        xmax = -1e30;
        ymin =  1e30;
        ymax = -1e30;
        zmin =  1e30;
        zmax = -1e30;
        ReadFromFile(filename);
        WriteLine(1, "Done");
    }
    
    void StlConverter::Allocate(int num)
    {
        normalData = (float*)Cmf_Alloc(3*num*sizeof(float));
        vertexData = (float*)Cmf_Alloc(9*num*sizeof(float));
        requireFree = true;
    }
    
    void StlConverter::SavePointCloud(std::string filename)
    {
        std::ofstream myfile;
		myfile.open (filename);
        for (int i = 0; i < 3*facetCount; i++)
        {
            myfile << vertexData[3*i+0] << ", " << vertexData[3*i+1] << ", " << vertexData[3*i+2] << "\n";
        }
        myfile.close();
    }
    
    void StlConverter::ReadFromFile(std::string filename)
    {
        if (!FileExists(filename)) CmfError("Cannot find file \"" + filename + "\".");
        FILE* fileReader;
        fileReader = fopen(filename.c_str(), "r+b");
        size_t dummy2;
        dummy2 = fread(header, sizeof(char), STL_HDR_SIZE, fileReader);
        dummy2 = fread(&facetCount, sizeof(int), 1, fileReader);
        Allocate(facetCount);

        char dummy[2];
        for (int i = 0; i < facetCount; i++)
        {
            dummy2 = fread(normalData+3*i, sizeof(float), 3, fileReader);
            dummy2 = fread(vertexData+9*i, sizeof(float), 9, fileReader);
            dummy2 = fread(dummy, sizeof(char), 2, fileReader);

            xmax = (vertexData[9*i+0]>xmax)?vertexData[9*i+0]:xmax;
            xmax = (vertexData[9*i+3]>xmax)?vertexData[9*i+3]:xmax;
            xmax = (vertexData[9*i+6]>xmax)?vertexData[9*i+6]:xmax;
            xmin = (vertexData[9*i+0]<xmin)?vertexData[9*i+0]:xmin;
            xmin = (vertexData[9*i+3]<xmin)?vertexData[9*i+3]:xmin;
            xmin = (vertexData[9*i+6]<xmin)?vertexData[9*i+6]:xmin;
            ymax = (vertexData[9*i+1]>ymax)?vertexData[9*i+1]:ymax;
            ymax = (vertexData[9*i+4]>ymax)?vertexData[9*i+4]:ymax;
            ymax = (vertexData[9*i+7]>ymax)?vertexData[9*i+7]:ymax;
            ymin = (vertexData[9*i+1]<ymin)?vertexData[9*i+1]:ymin;
            ymin = (vertexData[9*i+4]<ymin)?vertexData[9*i+4]:ymin;
            ymin = (vertexData[9*i+7]<ymin)?vertexData[9*i+7]:ymin;
            zmax = (vertexData[9*i+2]>zmax)?vertexData[9*i+2]:zmax;
            zmax = (vertexData[9*i+5]>zmax)?vertexData[9*i+5]:zmax;
            zmax = (vertexData[9*i+8]>zmax)?vertexData[9*i+8]:zmax;
            zmin = (vertexData[9*i+2]<zmin)?vertexData[9*i+2]:zmin;
            zmin = (vertexData[9*i+5]<zmin)?vertexData[9*i+5]:zmin;
            zmin = (vertexData[9*i+8]<zmin)?vertexData[9*i+8]:zmin;
        }
        fclose(fileReader);
    }
    
    void StlConverter::ConvertGeometry(SurfaceTriangulation* target)
    {
        WriteLine(3, "Converting to native geometry");
        target->AllocatePointBuffer(9*facetCount*sizeof(double));
        target->AllocateNormalBuffer(3*facetCount*sizeof(double));
        target->SetNumFaces(facetCount);
        //This is a temporary fix, will need to do duplicate elimination
        for (size_t i = 0; i < 9*facetCount; i++)
        {
            target->points[i] = 100000.00000;
        }
        for (size_t i = 0; i < 9*facetCount; i++)
        {
            target->points[i] = (double)vertexData[i];
        }
        for (size_t i = 0; i < 3*facetCount; i++)
        {
            target->normals[i] = (double)normalData[i];
        }
    }
    
    StlConverter::~StlConverter(void)
    {
        if (requireFree)
        {
            WriteLine(3, "Free stl resources");
            Cmf_Free(normalData);
            Cmf_Free(vertexData);
        }
    }
}