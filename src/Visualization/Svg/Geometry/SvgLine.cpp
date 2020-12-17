#include "SvgLine.h"
#include "SvgImage.h"
namespace cmf
{
    SvgLine::SvgLine(double x0, double y0, double x1, double y1, SvgImage* host)
    {
        host->MapPoint(x0, y0, &(coords[0]), &(coords[1]));
        host->MapPoint(x1, y1, &(coords[2]), &(coords[3]));
        title = "line";
    }
    
    void SvgLine::CreateAttributes(void)
    {
        attributes.Add("x1", coords[0]);
        attributes.Add("y1", coords[1]);
        attributes.Add("x2", coords[2]);
        attributes.Add("y2", coords[3]);
        CreateInheritedAttributes(attributes);
    }
    
    SvgLine::~SvgLine(void)
    {
        
    }
}