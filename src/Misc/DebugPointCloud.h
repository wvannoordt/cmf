#ifndef CMF_DEBUG_POINT_COULD_H
#define CMF_DEBUG_POINT_COULD_H

#include "Vec.h"

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

namespace cmf
{
    /// @brief A point cloud that can be used for debugging, generally a very slow implementation
    /// @author WVN
    struct DebugPointCloud
    {
        /// @brief Constructor
        /// @author WVN
        DebugPointCloud(void)
        {
            
        }
        
        /// @brief Adds a point to the point cloud
        /// @param point the point to add to the list
        /// @author WVN
        DebugPointCloud& operator << (Vec3<double> point)
        {
            points.push_back(point);
            return *this;
        }
        
        /// @brief Removes all points from the point cloud
        /// @author WVN
        void Clear(void)
        {
            points.clear();
        }
        
        void WriteVtk(std::string filename)
        {
            std::ofstream myfile;
            myfile.open(filename.c_str());
            
            myfile << "# vtk DataFile Version 3.0\npoint cloud\nASCII\nDATASET POLYDATA\nPOINTS " << points.size() << " double\n";
            for (auto& p:points) myfile << p[0] << " " << p[1] << " " << p[2] << "\n";
            myfile << "POINT_DATA " << points.size() << "\nSCALARS Node double\nLOOKUP_TABLE default\n";
            for (int i = 0; i < points.size(); i++) myfile << (double)i << "\n";
            myfile.close();
        }
        
        /// @brief List of points
        std::vector<Vec3<double>> points;
    };
}

#endif