#include "VtkTopology.h"
#include <iostream>
namespace Anaptric
{
    VtkTopology::VtkTopology(VtkTopologyType::VtkTopologyType topologyType_in)
    {
        elementType = "topology";
        topologyType = topologyType_in;
        deallocPointBuffer = false;
        deallocEdgeBuffer = false;
        numPointsVec[0] = -1;
        numPointsVec[1] = -1;
        numPointsVec[2] = -1;
        pointIndex = 0;
        edgeIndex = 0;
    }

    VtkTopology::VtkTopology(void)
    {
        elementType = "topology";
        topologyType = VtkTopologyType::polydata;
        deallocPointBuffer = false;
        deallocEdgeBuffer = false;
        numPointsVec[0] = -1;
        numPointsVec[1] = -1;
        numPointsVec[2] = -1;
        pointIndex = 0;
        edgeIndex = 0;
    }

    VtkTopology::~VtkTopology(void)
    {
        if (deallocPointBuffer) free(pointBuf);
        if (deallocEdgeBuffer) free(edgeBuf);
    }

    void VtkTopology::AddPoint(double x, double y)
    {
        AddPoint(x, y, 0.0);
    }

    void VtkTopology::AddPoint(double x, double y, double z)
    {
        *(pointBuf+3*pointIndex+0) = x;
        *(pointBuf+3*pointIndex+1) = y;
        *(pointBuf+3*pointIndex+2) = z;
        pointIndex++;
    }

    void VtkTopology::SetPointCount(size_t pointNum)
    {
        deallocPointBuffer = true;
        pointBuf = (double*)malloc(3*pointNum*sizeof(double));
        pointSize = pointNum;
        numPointsVec[0] = pointNum;
    }

    void VtkTopology::SetPointCount(size_t pointNumX, size_t pointNumY, size_t pointNumZ)
    {
        deallocPointBuffer = true;
        pointBuf = (double*)malloc(3*pointNumX*pointNumY*pointNumZ*sizeof(double));
        pointSize = pointNumX*pointNumY*pointNumZ;
        numPointsVec[0] = pointNumX;
        numPointsVec[1] = pointNumY;
        numPointsVec[2] = pointNumZ;
    }

    void VtkTopology::SetCellCount(size_t numCells, size_t numEntries)
    {
        deallocEdgeBuffer = true;
    }

    void VtkTopology::WriteToFile(std::ofstream & myfile)
    {
        myfile << "DATASET " << VtkTopologyTypeStr(topologyType) << std::endl;
        if (numPointsVec[1]>0 && numPointsVec[2]>0)
        {
            myfile << "POINS " << pointSize << " float" << std::endl;
        }
        else
        {
            myfile << "POINS " << numPointsVec[0] << " " << numPointsVec[1] << " " << numPointsVec[2] << " " << " float" << std::endl;
        }
        for (size_t idx = 0; idx < pointSize; idx++)
        {
            myfile << *(pointBuf+3*idx) << " " << *(pointBuf+3*idx+1) << " "<< *(pointBuf+3*idx+2) << std::endl;
        }

    }

    void VtkTopology::ReadFromFile(std::ofstream & myfile)
    {

    }
}
