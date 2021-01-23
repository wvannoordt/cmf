#ifndef CMF_SVG_ELEM_HANDLER_H
#define CMF_SVG_ELEM_HANDLER_H
#include "SvgElement.h"
#include "SvgAttributes.h"
#include "SvgLine.h"
#include "SvgRectangle.h"
namespace cmf
{
    class SvgManipulateItem;
    class SvgElementGroup;
    class SvgImage;
    /// @brief Class that represents an object that can create and manipulate SvgElement objects
	/// @author WVN
    class SvgElementHandler
    {
        public:
            /// @brief Constructor
        	/// @author WVN
            SvgElementHandler(void);
            
            /// @brief Destructor
        	/// @author WVN
            ~SvgElementHandler(void);
            
            /// @brief Creates a line
            /// @param start The starting point
            /// @param end The ending point
        	/// @author WVN
            SvgLine* AddLine(SvgNode* start, SvgNode* end);
            
            /// @brief Creates a rectangle
            /// @param lowerLeft The lower-left point
            /// @param upperRight The upper-right point
        	/// @author WVN
            SvgRectangle* AddRectangle(SvgNode* lowerLeft, SvgNode* upperRight);
            
        protected:
            
            /// @brief Sets the group that this object acts on
            /// @param layer_in The group to act on
        	/// @author WVN
            void BindGroup(SvgElementGroup* layer_in);
            
            /// @brief Sets the image that this object acts on
            /// @param image_in The group to act on
        	/// @author WVN
            void BindImage(SvgImage* image_in);
            
            /// @brief Throws an error if there is no group bound to this object
        	/// @author WVN
            void RequireBound(void);
            
            /// @brief Throws an error if there is no image bound to this object
        	/// @author WVN
            void RequireImage(void);
            
            /// @brief The group that this handler places and manipulated objects on
            SvgElementGroup* layer;
            
            /// @brief The image that the handler acts on
            SvgImage* image;
            
            /// @brief Indicates whether or not a group has been bound to this object
            bool bound;
            
            /// @brief Indicates whether or not an image has been bound to this object
            bool boundImage;
            
            /// @brief The layer/element attributes
            SvgAttributes attributes;
    };
}

#endif