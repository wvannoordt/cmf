#ifndef CMF_SVG_ELEM_HANDLER_H
#define CMF_SVG_ELEM_HANDLER_H
#include "SvgElement.h"
#include "SvgAttributes.h"
#include "SvgLine.h"
#include "SvgRectangle.h"
namespace cmf
{
    
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
            /// @param x0 x-coordinate of initial point
            /// @param x1 x-coordinate of final point
            /// @param y0 y-coordinate of initial point
            /// @param y1 y-coordinate of final point
        	/// @author WVN
            SvgLine* AddLine(double x0, double y0, double x1, double y1);
            
            /// @brief Creates a rectangle
            /// @param x0 x-coordinate of initial point
            /// @param x1 x-coordinate of final point
            /// @param y0 y-coordinate of initial point
            /// @param y1 y-coordinate of final point
        	/// @author WVN
            SvgRectangle* AddRectangle(double x0, double y0, double x1, double y1);
            
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