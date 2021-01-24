#include "SvgElementHandler.h"
#include "CmfError.h"
#include "SvgElementGroup.h"
namespace cmf
{
    SvgElementHandler::SvgElementHandler(void)
    {
        layer = NULL;
        bound = false;
        boundImage = false;
        image = NULL;
    }
    
    SvgElementHandler::~SvgElementHandler(void)
    {
        
    }
    
    void SvgElementHandler::RequireBound(void)
    {
        if (!bound) CmfError("This handler object has no group bound to it!");
    }
    
    void SvgElementHandler::RequireImage(void)
    {
        if (!boundImage) CmfError("This handler object has no image bound to it!");
    }
    
    SvgLine* SvgElementHandler::AddLine(SvgNode* start, SvgNode* end)
    {
        RequireBound();
        RequireImage();
        SvgLine* newLine = new SvgLine(start, end, image);
        layer->AddElementAndInternallyManage(newLine);
        return newLine;
    }
    
    SvgRectangle* SvgElementHandler::AddRectangle(SvgNode* lowerLeft, SvgNode* upperRight)
    {
        RequireBound();
        RequireImage();
        SvgRectangle* newRect = new SvgRectangle(lowerLeft, upperRight, image);
        layer->AddElementAndInternallyManage(newRect);
        return newRect;
    }
    
    SvgCircle* SvgElementHandler::AddCircle(SvgNode* center, double radius)
    {
        RequireBound();
        RequireImage();
        SvgCircle* newCirc = new SvgCircle(center, radius, image);
        layer->AddElementAndInternallyManage(newCirc);
        return newCirc;
    }
    
    void SvgElementHandler::BindGroup(SvgElementGroup* layer_in)
    {
        bound = true;
        layer = layer_in;
    }
    
    void SvgElementHandler::BindImage(SvgImage* image_in)
    {
        boundImage = true;
        image = image_in;
    }
}