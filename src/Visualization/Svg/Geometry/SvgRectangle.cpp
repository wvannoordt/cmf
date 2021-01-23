#include "SvgRectangle.h"
#include "SvgImage.h"
#include <cmath>
namespace cmf
{
    SvgRectangle::SvgRectangle(SvgNode* lowerLeft_in, SvgNode* upperRight_in, double rx, double ry, SvgImage* host)
    {
        Build(lowerLeft_in, upperRight_in, rx, ry, host);
    }
    
    SvgRectangle::SvgRectangle(SvgNode* lowerLeft_in, SvgNode* upperRight_in, SvgImage* host)
    {
        Build(lowerLeft_in, upperRight_in, 0.0, 0.0, host);
    }
    
    SvgRectangle::~SvgRectangle(void)
    {
        
    }
    
    void SvgRectangle::Build(SvgNode* lowerLeft_in, SvgNode* upperRight_in, double rx, double ry, SvgImage* host)
    {
        width = upperRight_in->x - lowerLeft_in->x;
        height = upperRight_in->y - lowerLeft_in->y;
        lowerLeft = lowerLeft_in;
        upperRight = upperRight_in;
        title = "rect";
        roundx = rx;
        roundy = ry;
        hostImage = host;
    }
    
    void SvgRectangle::CreateAttributes(void)
    {
        double x = 0.0;
        double y = 0.0;
        this->GetImage()->MapPoint(lowerLeft->x, upperRight->y, &x, &y);
        attributes.Add("x", x);
        attributes.Add("y", y);
        attributes.Add("width", width);
        attributes.Add("height", height);
        if (abs(roundx) > 1e-9) attributes.Add("rx", roundx);
        if (abs(roundy) > 1e-9) attributes.Add("ry", roundy);
        SvgHasFill::CreateInheritedAttributes(attributes);
        SvgHasStroke::CreateInheritedAttributes(attributes);
    }
}