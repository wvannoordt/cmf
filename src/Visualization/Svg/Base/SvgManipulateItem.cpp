#include "SvgManipulateItem.h"
#include "SvgElementHandler.h"
namespace cmf
{
    SvgManipulateItem::SvgManipulateItem(void)
    {
        visible = true;
        containerPosition = -1;
    }
    SvgManipulateItem::~SvgManipulateItem(void)
    {
        
    }
    bool SvgManipulateItem::SetVisibility(bool val)
    {
        visible = val;
    }
    bool SvgManipulateItem::IsVisible(void)
    {
        return visible;
    }
}