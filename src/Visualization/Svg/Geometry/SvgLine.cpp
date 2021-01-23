#include "SvgLine.h"
#include "SvgImage.h"
namespace cmf
{
    SvgLine::SvgLine(SvgNode* startNode_in, SvgNode* endNode_in, SvgImage* host)
    {
        startNode = startNode_in;
        endNode = endNode_in;
        title = "line";
        hostImage = host;
    }
    
    void SvgLine::CreateAttributes(void)
    {
        double x1, x2, y1, y2;
        this->GetImage()->MapPoint(startNode->x, startNode->y, &x1, &y1);
        this->GetImage()->MapPoint(endNode->x,   endNode->y,   &x2, &y2);
        attributes.Add("x1", x1);
        attributes.Add("y1", y1);
        attributes.Add("x2", x2);
        attributes.Add("y2", y2);
        CreateInheritedAttributes(attributes);
    }
    
    SvgLine::~SvgLine(void)
    {
        
    }
}