#ifndef CMF_SVG_HAS_FILL_H
#define CMF_SVG_HAS_FILL_H
#include "SvgColor.h"
#include "SvgAttributeModifier.h"
namespace cmf
{
    /// @brief Struct containing information for a line stroke
	/// @author WVN
    struct SvgFill
    {
        /// @brief Default constructor
    	/// @author WVN
        SvgFill(void)
        {
            fillOpacity = 1.0;
            fillColor = "black";
        }
        
        /// @brief Copies values from another instance
        /// @param f The value to copy
    	/// @author WVN
        void Copy(SvgFill& f)
        {
            fillOpacity = f.fillOpacity;
            fillColor.Copy(f.GetColor());
        }
        
        /// @brief Sets the opacity value
        /// @param val The value to set
        /// @author WVN
        void SetOpacity(double val)
        {
            fillOpacity = val;
        }
        
        /// @brief Sets the fill color
        /// @param val The value to set
        /// @author WVN
        void SetColor(std::string val)
        {
            fillColor = val;
        }
        
        /// @brief Sets the fill color
        /// @param val The value to set
        /// @author WVN
        void SetColor(double val[3])
        {
            fillColor = val;
        }
        
        /// @brief Returns the color object
        /// @author WVN
        SvgColor* GetColor(void)
        {
            return &fillColor;
        }
        
        /// @brief Width of the fill
        double fillOpacity;
        
        /// @brief Color of the fill
        SvgColor fillColor;
    };
    
    /// @brief Class to inherit when an SvgElement has a fill property
	/// @author WVN
    class SvgHasFill : public SvgAttributeModifier
    {
        public:
            /// @brief Empty constructor
        	/// @author WVN
            SvgHasFill(void){}
            
            /// @brief Empty destructor
        	/// @author WVN
            ~SvgHasFill(void){}
            
            /// @brief Overwrites the fill properties
            /// @param fillProps The fill properties to overwrite
        	/// @author WVN
            void SetFill(SvgFill& fillProps)
            {
                fill.Copy(fillProps);
            }
            
            /// @brief Sets the opacity value
            /// @param val The value to set
        	/// @author WVN
            void SetFillOpacity(double val)
            {
                fill.SetOpacity(val);
            }
            
            /// @brief Sets the fill color
            /// @param val The value to set
        	/// @author WVN
            void SetFillColor(double val[3])
            {
                fill.SetColor(val);
            }
            
            /// @brief Sets the fill color
            /// @param val The value to set
        	/// @author WVN
            void SetFillColor(std::string val)
            {
                fill.SetColor(val);
            }
            
            /// @brief gets the fill color
        	/// @author WVN
            SvgColor* GetFillColor(void)
            {
                return fill.GetColor();
            }
            
            /// @brief gets the fill object
        	/// @author WVN
            SvgFill* GetFill(void)
            {
                return &fill;
            }
            
            /// @brief Fills the attribute object
        	/// @author WVN
            void CreateInheritedAttributes(SvgAttributes& attr) final
            {
                attr.Add("fill", fill.GetColor()->StringValue());
                attr.Add("fill-opacity", fill.fillOpacity);
            }
        
        protected:
            
            /// @brief The fill properties
            SvgFill fill;
    };
}

#endif