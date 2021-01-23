#ifndef CMF_SVG_RECT_H
#define CMF_SVG_RECT_H
#include "SvgElement.h"
#include "SvgHasStroke.h"
#include "SvgHasFill.h"
#include "SvgNode.h"
namespace cmf
{
    /// @brief Class that represents a rectangle on an svg image
	/// @author WVN
    class SvgRectangle : public SvgElement, public SvgHasStroke, public SvgHasFill
    {
        public:
            /// @brief Constructor
            /// @param lowerLeft_in The lower-left point
            /// @param upperRight_in The upper-right point
            /// @param rx x-Round parameter for rounding corners
            /// @param ry y-Round parameter for rounding corners
        	/// @author WVN
            SvgRectangle(SvgNode* lowerLeft_in, SvgNode* upperRight_in, double rx, double ry, SvgImage* host);
            
            /// @brief Constructor
            /// @param lowerLeft_in The lower-left point
            /// @param upperRight_in The upper-right point
        	/// @author WVN
            SvgRectangle(SvgNode* lowerLeft_in, SvgNode* upperRight_in, SvgImage* host);
            
            /// @brief Creates the relevant attributes for file output
        	/// @author WVN
            void CreateAttributes(void);
            
            /// @brief Denstructor
        	/// @author WVN
            ~SvgRectangle(void);
            
        private:
            
            /// @brief Builder function
            /// @param lowerLeft_in The lower-left point
            /// @param upperRight_in The upper-right point
            /// @param rx x-Round parameter for rounding corners
            /// @param ry y-Round parameter for rounding corners
        	/// @author WVN
            void Build(SvgNode* lowerLeft_in, SvgNode* upperRight_in, double rx, double ry, SvgImage* host);
            
            /// @brief The lower-left point
            SvgNode* lowerLeft;
            
            /// @brief The upper-right point
            SvgNode* upperRight;
            
            /// @brief rounding parameter, x
            double roundx;
            
            /// @brief rounding parameter, y
            double roundy;
            
            /// @brief rectangle width
            double width;
            
            /// @brief rectangle height
            double height;
    };
}

#endif