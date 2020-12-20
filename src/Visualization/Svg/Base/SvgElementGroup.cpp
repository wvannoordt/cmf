#include "SvgElementGroup.h"
#include "SvgImage.h"
#include "SvgAttributes.h"
namespace cmf
{
    SvgElementGroup::SvgElementGroup(std::string groupName_in, SvgImage* hostImage_in)
    {
        groupName = groupName_in;
        BindGroup(this);
        BindImage(hostImage_in);
        attributes.Add("id", groupName);
        hostImage = hostImage_in;
        isGroup = true;
    }
    
    void SvgElementGroup::WriteToFile(SvgFileStream& stream)
    {
        // SvgAttributes transform;
        // transform.Add("transform", "matrix(1 0 0 -1 0 " + std::to_string(hostImage->GetHeight()) + ")");
        stream.AddLine("<g " + attributes.GetString() + ">");
        stream.Indent();
        for (int i = 0; i < elements.size(); i++)
        {
            if (elements[i]->IsVisible())
            {
                elements[i]->WriteToFile(stream);
            }
        }
        stream.UnIndent();
        stream.AddLine("</g>");
    }
    
    void SvgElementGroup::BringForward(void)
    {
        SvgElementGroup* swap;
        std::string sSwap;
        std::vector<SvgElementGroup*>& elements = hostImage->GetGroups();
        std::vector<std::string>& names = hostImage->GetGroupNames();
        int myPosition = containerPosition;
        if (myPosition == (elements.size()-1)) return;
        swap = elements[myPosition+1];
        sSwap = names[myPosition+1];
        elements[myPosition+1] = this;
        elements[myPosition] = swap;
        names[myPosition+1] = this->groupName;
        names[myPosition] = sSwap;
        elements[myPosition+1]->containerPosition = myPosition + 1;
        elements[myPosition]->containerPosition = myPosition;
    }
    void SvgElementGroup::BringToFront(void)
    {
        std::vector<SvgElementGroup*>& elements = hostImage->GetGroups();
        std::vector<std::string>& names = hostImage->GetGroupNames();
        int myPosition = containerPosition;
        if (myPosition == (elements.size()-1)) return;
        elements.push_back(this);
        elements.erase(elements.begin() + myPosition);
        names.push_back(this->groupName);
        names.erase(names.begin() + myPosition);
        for (int i = myPosition; i < elements.size(); i++)
        {
            elements[i]->containerPosition = i;
        }
    }
    void SvgElementGroup::SendBackward(void)
    {
        SvgElementGroup* swap;
        std::string sSwap;
        std::vector<SvgElementGroup*>& elements = hostImage->GetGroups();
        std::vector<std::string>& names = hostImage->GetGroupNames();
        int myPosition = containerPosition;
        if (myPosition == (0)) return;
        swap = elements[myPosition-1];
        sSwap = names[myPosition-1];
        elements[myPosition-1] = this;
        elements[myPosition] = swap;
        names[myPosition-1] = this->groupName;
        names[myPosition] = sSwap;
        elements[myPosition-1]->containerPosition = myPosition - 1;
        elements[myPosition]->containerPosition = myPosition;
    }
    void SvgElementGroup::SendToBack(void)
    {
        std::vector<SvgElementGroup*>& elements = hostImage->GetGroups();
        std::vector<std::string>& names = hostImage->GetGroupNames();
        int myPosition = containerPosition;
        if (myPosition == 0) return;
        elements.insert(elements.begin(), this);
        elements.erase(elements.begin() + myPosition);
        names.insert(names.begin(), this->groupName);
        names.erase(names.begin() + myPosition);
        for (int i = 0; i <= myPosition; i++)
        {
            elements[i]->containerPosition = i;
        }
    }
    
    SvgElementGroup::~SvgElementGroup(void)
    {
        for (int i = 0; i < elements.size(); i++)
        {
            if ((elements[i]->isExternallyCreated))
            {
                delete elements[i];
            }
        }
    }
}