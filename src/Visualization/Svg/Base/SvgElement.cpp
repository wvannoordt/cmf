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
        isGroup = false;
    }
    
    void SvgElement::SetGroup(SvgElementGroup* gr)
    {
        containingGroup = gr;
    }
    
    void SvgElement::SetImage(SvgImage* im)
    {
        hostImage = im;
    }
        
    void SvgElement::BringForward(void)
    {
        SvgElement* swap;
        std::vector<SvgElement*>& elements = *(&(containingGroup->elements));
        int myPosition = containerPosition;
        if (myPosition == (elements.size()-1)) return;
        swap = elements[myPosition+1];
        elements[myPosition+1] = this;
        elements[myPosition] = swap;
        elements[myPosition+1]->containerPosition = myPosition + 1;
        elements[myPosition]->containerPosition = myPosition;
    }
    void SvgElement::BringToFront(void)
    {
        std::vector<SvgElement*>& elements = *(&(containingGroup->elements));
        int myPosition = containerPosition;
        if (myPosition == (elements.size()-1)) return;
        elements.push_back(this);
        elements.erase(elements.begin() + myPosition);
        for (int i = myPosition; i < elements.size(); i++)
        {
            elements[i]->containerPosition = i;
        }
    }
    void SvgElement::SendBackward(void)
    {
        SvgElement* swap;
        std::vector<SvgElement*>& elements = *(&(containingGroup->elements));
        int myPosition = containerPosition;
        if (myPosition == (0)) return;
        swap = elements[myPosition-1];
        elements[myPosition-1] = this;
        elements[myPosition] = swap;
        elements[myPosition-1]->containerPosition = myPosition - 1;
        elements[myPosition]->containerPosition = myPosition;
    }
    void SvgElement::SendToBack(void)
    {
        std::vector<SvgElement*>& elements = *(&(containingGroup->elements));
        int myPosition = containerPosition;
        if (myPosition == 0) return;
        elements.erase(elements.begin() + myPosition);
        elements.insert(elements.begin(), this);
        for (int i = 0; i <= myPosition; i++)
        {
            elements[i]->containerPosition = i;
        }
    }
    
    SvgElement::~SvgElement(void)
    {
        
    }
}