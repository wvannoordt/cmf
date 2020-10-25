#ifndef CMF_AXISALIGNEDLONGCYLINDER
#define CMF_AXISALIGNEDLONGCYLINDER
#include "ICmfGeometry.h"

namespace cmf
{
    /// @brief A class representing an infinitely long cylinder aligned with a given Cartesian direction
    /// @author WVN
    class AxisAlignedLongCylinder : ICmfGeometry
    {
        public:
            /// @brief The constructor for the AxisAlignedLongCylinder class
            /// @param point_in The coordinates of a reference point (size CMF_DIM) along the axis of the cylinder
            /// @param radius_in The radius of the cylinder
            /// @param axisAlignment_in The direction of axis alignment (0, 1, [2])
            /// @author WVN
            AxisAlignedLongCylinder(double* point_in, double radius_in, int axisAlignment_in);
            
            /// @brief The destructor for the AxisAlignedLongCylinder class
            /// @author WVN
            ~AxisAlignedLongCylinder(void);
            
            /// @brief See ICmfGeometry for documentation
            /// @author WVN
            bool BoxIntersectsBoundary(double* bounds);
            
        private:
            
            /// @brief Checks if a circle intersects the edge of a box
            /// @param c Center coordinates
            /// @param x1 x-coordinate of first edge point
            /// @param y1 y-coordinate of first edge point
            /// @param x2 x-coordinate of second edge point
            /// @param y2 y-coordinate of second edge point
            /// @param r Radius of circle
            /// @author WVN
            bool CheckEdge(double* c, double x1, double y1, double x2, double y2, double r);
            
            /// @brief Checks if a circle intersects a box
            /// @param c Center coordinates
            /// @param r Radius of circle
            /// @param box bounding box to check
            /// @author WVN
            bool CircleIntersects(double* c, double r, double* box);
            
            /// @brief The coordinates of a reference point (size CMF_DIM) along the axis of the cylinder
            double point[2];
            
            /// @brief The radius of the cylinder
            double radius;
            
            /// @brief The direction of axis alignment (0, 1, [2])
            int axisAlignment;
    };
}
#endif