#include "AxisAlignedLongCylinder.h"
#include "Utils.hx"
namespace cmf
{
    AxisAlignedLongCylinder::AxisAlignedLongCylinder(double* point_in, double radius_in, int axisAlignment_in)
    {
        int idx = 0;
        point[0] = 0.0;
        point[1] = 0.0;
        axisAlignment = axisAlignment_in;
        for (int d = 0; d < CMF_DIM; d++)
        {
            if (d!=axisAlignment)
            {
                point[idx]=point_in[d];
                idx++;
            }
        }
        radius = radius_in;
        geometryType = "AxisAlignedLongCylinder";
    }
    
    AxisAlignedLongCylinder::~AxisAlignedLongCylinder(void)
    {
        
    }
    
    bool AxisAlignedLongCylinder::BoxIntersectsBoundary(double* bounds)
    {
        double box2D[4]={0.0,0.0,0.0,0.0};
        int idx = 0;
        __dloop(if (d!=axisAlignment) {box2D[2*idx]=bounds[2*d];box2D[2*idx+1]=bounds[2*d+1];idx++;});
        return CircleIntersects(point, radius, box2D);
    }
    
    bool AxisAlignedLongCylinder::CircleIntersects(double* c, double r, double* box)
    {
        bool negativeFound = false;
        bool positiveFound = false;
        double corner[2];
        for (int i = 0; i < 4; i++)
        {
            corner[0] = (i&1)*box[0] + (1 - (i&1))*box[1];
            corner[1] = ((i&2)/2)*box[2] + (1-((i&2)/2))*box[3];
            double q = (c[0]-corner[0])*(c[0]-corner[0]) + (c[1]-corner[1])*(c[1]-corner[1]) - r*r;
            negativeFound = negativeFound || (q<0);
            positiveFound = positiveFound || (q>0);
        }
        bool aCornerIsIn = (positiveFound && negativeFound);
        bool isContained = true;
        isContained = isContained && ((c[0] - r) > box[0]);
        isContained = isContained && ((c[0] + r) < box[1]);
        isContained = isContained && ((c[1] - r) > box[2]);
        isContained = isContained && ((c[1] + r) < box[3]);
        bool anEdgeIsIn = false;
        anEdgeIsIn = anEdgeIsIn || (CheckEdge(c, box[0], box[2], box[0], box[3], r));
        anEdgeIsIn = anEdgeIsIn || (CheckEdge(c, box[1], box[2], box[1], box[3], r));
        anEdgeIsIn = anEdgeIsIn || (CheckEdge(c, box[0], box[2], box[1], box[2], r));
        anEdgeIsIn = anEdgeIsIn || (CheckEdge(c, box[0], box[3], box[1], box[3], r));
        return aCornerIsIn || isContained || anEdgeIsIn;
    }
    
    bool AxisAlignedLongCylinder::CheckEdge(double* c, double x1, double y1, double x2, double y2, double r)
    {
        double x0 = c[0];
        double y0 = c[1];
        double a = (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1;
        if (a<0) a = -a;
        double d = sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1));
        double alpha = (x1-x2)*(x1-x0) + (y1-y2)*(y1-y0);
        return (a/d < r) && ((sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0))>r)||(sqrt((x2-x0)*(x2-x0) + (y2-y0)*(y2-y0))>r)) && (0 <= alpha) && (alpha <= d*d);
    }
}