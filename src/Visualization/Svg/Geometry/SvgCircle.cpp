#include "SvgCircle.h"
#include "SvgImage.h"
namespace cmf
{
    SvgCircle::SvgCircle(SvgNode* center_in, double radius_in, SvgImage* hostImage_in)
    {
        center = center_in;
        radius = radius_in;
        hostImage = hostImage_in;
        title = "circle";
    }
    
    void SvgCircle::CreateAttributes(void)
    {
        double x = 0.0;
        double y = 0.0;
        this->GetImage()->MapPoint(center->x, center->y, &x, &y);
        attributes.Add("cx", x);
        attributes.Add("cy", y);
        attributes.Add("r", radius);
        SvgHasFill::CreateInheritedAttributes(attributes);
        SvgHasStroke::CreateInheritedAttributes(attributes);
    }
    
    SvgCircle::~SvgCircle()
    {
        
    }
}