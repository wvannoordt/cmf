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