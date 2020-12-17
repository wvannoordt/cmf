#include "SvgElement.h"
#include "SvgImage.h"
#include "SvgElementGroup.h"
namespace cmf
{
    SvgElement::SvgElement(void)
    {
        isExternallyCreated = false;
        hostImage = NULL;
        containingGroup = NULL;
    }
    
    void SvgElement::SetGroup(SvgElementGroup* gr)
    {
        containingGroup = gr;
    }
    
    void SvgElement::SetImage(SvgImage* im)
    {
        hostImage = im;
    }
    
    SvgElement::~SvgElement(void)
    {
        
    }
}