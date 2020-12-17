#ifndef CMF_SVG_RECT_H
#define CMF_SVG_RECT_H
#include "SvgElement.h"
#include "SvgHasStroke.h"
#include "SvgHasFill.h"
namespace cmf
{
    /// @brief Class that represents a rectangle on an svg image
	/// @author WVN
    class SvgRectangle : public SvgElement, public SvgHasStroke, public SvgHasFill
    {
        public:
            /// @brief Constructor
            /// @param x0 x-coordinate of first corner
            /// @param x1 x-coordinate of first corner
            /// @param y0 y-coordinate of second corner
            /// @param y1 y-coordinate of second corner
            /// @param rx x-Round parameter for rounding corners
            /// @param ry y-Round parameter for rounding corners
        	/// @author WVN
            SvgRectangle(double x0, double y0, double x1, double y1, double rx, double ry, SvgImage* host);
            
            /// @brief Constructor
            /// @param x0 x-coordinate of first corner
            /// @param x1 x-coordinate of first corner
            /// @param y0 y-coordinate of second corner
            /// @param y1 y-coordinate of second corner
        	/// @author WVN
            SvgRectangle(double x0, double y0, double x1, double y1, SvgImage* host);
            
            /// @brief Creates the relevant attributes for file output
        	/// @author WVN
            void CreateAttributes(void);
            
            /// @brief Denstructor
        	/// @author WVN
            ~SvgRectangle(void);
            
        private:
            
            /// @brief Builder function
            /// @param x0 x-coordinate of first corner
            /// @param x1 x-coordinate of first corner
            /// @param y0 y-coordinate of second corner
            /// @param y1 y-coordinate of second corner
            /// @param rx x-Round parameter for rounding corners
            /// @param ry y-Round parameter for rounding corners
        	/// @author WVN
            void Build(double x0, double y0, double x1, double y1, double rx, double ry, SvgImage* host);
            
            /// @brief coordinates of the rectangle: (x0, y0, x1, y1)
            double coords[4];
            
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