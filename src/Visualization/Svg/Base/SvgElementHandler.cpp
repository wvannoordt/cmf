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
    
    SvgLine* SvgElementHandler::AddLine(double x0, double y0, double x1, double y1)
    {
        RequireBound();
        RequireImage();
        SvgLine* newLine = new SvgLine(x0, y0, x1, y1, image);
        layer->AddElementAndInternallyManage(newLine);
        return newLine;
    }
    
    SvgRectangle* SvgElementHandler::AddRectangle(double x0, double y0, double x1, double y1)
    {
        RequireBound();
        RequireImage();
        SvgRectangle* newRect = new SvgRectangle(x0, y0, x1, y1, image);
        layer->AddElementAndInternallyManage(newRect);
        return newRect;
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