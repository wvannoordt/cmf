#ifndef CMF_SVG_HAS_STROKE_H
#define CMF_SVG_HAS_STROKE_H
#include "SvgColor.h"
#include "SvgAttributeModifier.h"
namespace cmf
{
    /// @brief Struct containing information for a line stroke
	/// @author WVN
    struct SvgStroke
    {
        /// @brief Default constructor
    	/// @author WVN
        SvgStroke(void)
        {
            strokeWidth = 1.0;
            strokeColor = "black";
        }
        /// @brief Width of the stroke
        double strokeWidth;
        
        /// @brief Color of the stroke
        SvgColor strokeColor;
    };
    
    /// @brief Class to inherit when an SvgElement has a stroke property
	/// @author WVN
    class SvgHasStroke : public SvgAttributeModifier
    {
        public:
            /// @brief Empty constructor
        	/// @author WVN
            SvgHasStroke(void){}
            
            /// @brief Empty destructor
        	/// @author WVN
            ~SvgHasStroke(void){}
            
            
            /// @brief Fills the attribute object
        	/// @author WVN
            void CreateInheritedAttributes(SvgAttributes& attr) final
            {
                attr.Add("stroke", stroke.strokeColor.StringValue());
                attr.Add("stroke-width", stroke.strokeWidth);
            }
        
        private:
            
            /// @brief The stroke properties
            SvgStroke stroke;
    };
}

#endif