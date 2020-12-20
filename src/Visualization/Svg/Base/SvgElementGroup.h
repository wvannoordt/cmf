#ifndef CMF_SVG_ELEMENT_GROUP_H
#define CMF_SVG_ELEMENT_GROUP_H
#include <string>
#include "SvgElement.h"
#include "SvgElementHandler.h"
#include "SvgManipulateItem.h"
#include "SvgFileStream.h"
#include <vector>
namespace cmf
{
    class SvgImage;
    /// @brief Represents a group/layer of SVG elements
	/// @author WVN
    class SvgElementGroup : public SvgElementHandler, public SvgManipulateItem
    {
        public:
            /// @brief Constructor
            /// @param groupName_in The name of this group
        	/// @author WVN
            SvgElementGroup(std::string groupName_in, SvgImage* hostImage_in);
            
            /// @brief Destructor
        	/// @author WVN
            ~SvgElementGroup(void);
            
            /// @brief Writes the group to a file stream
            /// @param stream The stream to write to
        	/// @author WVN
            void WriteToFile(SvgFileStream& stream);
            
            /// @brief Returns the relevant image
            /// @author WVN
            SvgImage* GetImage(void){return hostImage;}
            
            /// @brief Returns this group
            /// @author WVN
            SvgElementGroup* GetGroup(void){return this;}
            
            
            /// @brief Advances the item one position within its container
            /// @author WVN
            void BringForward(void);
            
            /// @brief Advances the item to the front of its container
            /// @author WVN
            void BringToFront(void);
            
            /// @brief Retreats the item one position within its container
            /// @author WVN
            void SendBackward(void);
            
            /// @brief Sends the item to the first position within its container
            /// @author WVN
            void SendToBack(void);
            
            /// @brief Returns the lisst of groupd
            /// @author WVN
            std::vector<SvgElement*>& GetElements(void);
            
        private:
            /// @brief The name of this group
            std::string groupName;
            
            /// @brief The list of elements in this group
            std::vector<SvgElement*> elements;
            
            /// @brief The containing image
            SvgImage* hostImage;
            
            /// @brief Adds an element to the group and asserts that the group must later delete the element
            /// @param elem The element to add
        	/// @author WVN
            void AddElementAndInternallyManage(SvgElement* elem)
            {
                elem->SetImage(hostImage);
                elem->SetGroup(this);
                elem->SetPosition(elements.size());
                elem->isExternallyCreated = true;
                elements.push_back(elem);
            }
            friend class SvgElement;
            friend class SvgElementHandler;
    };
}

#endif