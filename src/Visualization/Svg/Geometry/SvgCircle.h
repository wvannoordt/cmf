#ifndef CMF_SVG_CIRCLE_H
#define CMF_SVG_CIRCLE_H
#include "SvgElement.h"
#include "SvgHasFill.h"
#include "SvgHasStroke.h"
#include "SvgNode.h"

namespace cmf
{
    /// @brief Class that represents a circle on an svg image
	/// @author WVN
    class SvgCircle : public SvgElement, public SvgHasStroke, public SvgHasFill
    {
        public:
            /// @brief Constructor
            /// @param center_in The center of the circle
            /// @param radius_in The radius of the circle
            /// @param hostImage_in The svg image to draw on
        	/// @author WVN
            SvgCircle(SvgNode* center_in, double radius_in, SvgImage* hostImage_in);
            
            /// @brief Destructor
        	/// @author WVN
            ~SvgCircle(void);
            
            /// @brief Creates the relevant attributes for file output
        	/// @author WVN
            void CreateAttributes(void) override;
            
        private:
            
            ///@brief The center of the circle
            SvgNode* center;
            
            ///@brief The radius of the circle
            double radius;
    };
}

#endif