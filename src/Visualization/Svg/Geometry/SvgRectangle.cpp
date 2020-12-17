#include "SvgRectangle.h"
#include "SvgImage.h"
#include <cmath>
namespace cmf
{
    SvgRectangle::SvgRectangle(double x0, double y0, double x1, double y1, double rx, double ry, SvgImage* host)
    {
        Build(x0, y0, x1, y1, rx, ry, host);
    }
    
    SvgRectangle::SvgRectangle(double x0, double y0, double x1, double y1, SvgImage* host)
    {
        Build(x0, y0, x1, y1, 0.0, 0.0, host);
    }
    
    SvgRectangle::~SvgRectangle(void)
    {
        
    }
    
    void SvgRectangle::Build(double x0, double y0, double x1, double y1, double rx, double ry, SvgImage* host)
    {
        width = x1 - x0;
        height = y1 - y0;
        host->MapPoint(x0, y0, &(coords[0]), &(coords[1]));
        host->MapPoint(x1, y1, &(coords[2]), &(coords[3]));
        title = "rect";
        roundx = rx;
        roundy = ry;
    }
    
    void SvgRectangle::CreateAttributes(void)
    {
        attributes.Add("x", coords[0]);
        attributes.Add("y", coords[3]);
        attributes.Add("width", width);
        attributes.Add("height", height);
        if (abs(roundx) > 1e-9) attributes.Add("rx", roundx);
        if (abs(roundy) > 1e-9) attributes.Add("ry", roundy);
        SvgHasFill::CreateInheritedAttributes(attributes);
        SvgHasStroke::CreateInheritedAttributes(attributes);
    }
}