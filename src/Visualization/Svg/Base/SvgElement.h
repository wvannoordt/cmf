#ifndef CMF_SVG_ELEMENT_H
#define CMF_SVG_ELEMENT_H
#include <ostream>
#include "SvgManipulateItem.h"
#include "SvgFileStream.h"
#include "SvgAttributes.h"
namespace cmf
{
    class SvgImage;
    class SvgElementGroup;
    /// @brief Represents a generic element in an SVG image
	/// @author WVN
    class SvgElement : public SvgManipulateItem
    {
        public:
            /// @brief Constructor
        	/// @author WVN
            SvgElement(void);
            
            /// @brief Destructor
        	/// @author WVN
            ~SvgElement(void);
            
            /// @brief Writes the contents of the element to a file stream
            /// @param myfile The stream to write to
        	/// @author WVN
            void WriteToFile(SvgFileStream& stream)
            {
                CreateAttributes();
                stream.AddLine("<" + title + " " + attributes.GetString() + "/>");
            }
            
            /// @brief Sets the host image
            /// @param im The image to set
        	/// @author WVN
            void SetImage(SvgImage* im);
            
            /// @brief Creates the relevant attributes for file output
        	/// @author WVN
            virtual void CreateAttributes(void) {}
            
            /// @brief Returns the relevant image
            /// @author WVN
            SvgImage* GetImage(void){return hostImage;}
            
            /// @brief Sets the containing group
            /// @param gr The group to set
        	/// @author WVN
            void SetGroup(SvgElementGroup* gr);
            
            /// @brief Returns the containing group
            /// @author WVN
            SvgElementGroup* GetGroup(void){return containingGroup;}
            
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
            
        protected:
            
            /// @brief The element attributes
            SvgAttributes attributes;
            
            /// @brief The element title
            std::string title;
            
            /// @brief The containing image
            SvgImage* hostImage;
            
            /// @brief The containing group
            SvgElementGroup* containingGroup;
        
        private:
            
            /// @brief Indicates whether or not the element was created by the user
            bool isExternallyCreated;
        
        friend class SvgElementGroup;
    };
}

#endif