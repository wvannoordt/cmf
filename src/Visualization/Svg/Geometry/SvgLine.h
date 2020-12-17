#ifndef CMF_SVG_LINE_H
#define CMF_SVG_LINE_H
#include "SvgElement.h"
#include "SvgHasStroke.h"
namespace cmf
{
    /// @brief Class that represents a line on an svg image
	/// @author WVN
    class SvgLine : public SvgElement, public SvgHasStroke
    {
        public:
            /// @brief Constructor
            /// @param x0 x-coordinate of initial point
            /// @param x1 x-coordinate of final point
            /// @param y0 y-coordinate of initial point
            /// @param y1 y-coordinate of final point
        	/// @author WVN
            SvgLine(double x0, double y0, double x1, double y1, SvgImage* host);
            
            /// @brief Creates the relevant attributes for file output
        	/// @author WVN
            void CreateAttributes(void);
            
            /// @brief Destructor
        	/// @author WVN
            ~SvgLine(void);
            
        private:
            /// @brief The coordinates of the line (x0 y0 x1 y1)
            double coords[4];
    };
}

#endif