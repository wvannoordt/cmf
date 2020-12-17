#ifndef CMF_SVG_COLOR_H
#define CMF_SVG_COLOR_H
#include <string>
namespace cmf
{
    /// @brief A class representing a color withing an SVG image
    /// @author WVN
    class SvgColor
    {
        public:
            /// @brief Default constructor
            /// @author WVN
            SvgColor(void){val = "black";}
            
            /// @brief Default destructor
            /// @author WVN
            ~SvgColor(void){}
            
            /// @brief Allows assigning based on  a string value
            /// @author WVN
            SvgColor& operator= (std::string val_r)
            {
                val = val_r;
                return *this;
            }
            
            /// @brief Allows assigning based on a double triplet
            /// @author WVN
            SvgColor& operator= (double val_r[3]) {val = "#" + NumberToHexValue(val_r[0])+NumberToHexValue(val_r[1])+NumberToHexValue(val_r[2]);}
            
            /// @brief Returns val
            /// @author WVN
            std::string StringValue(void)
            {
                return val;
            }
            
            /// @brief Copies values from another instance
            /// @param col The value to copy
        	/// @author WVN
            void Copy(SvgColor* col)
            {
                val = col->val;
            }
            
            /// @brief Returns a pointer to the string value (for debugging only)
        	/// @author WVN
            std::string* GetValuePointer(void)
            {
                return &val;
            }
            
        private:
            /// @brief a string value that will finally be written to an attribute
            std::string val;
            
            /// @brief Returns a hexadecimal value from a double between 0 and 1
            /// @param d a value between 0 and 1;
            /// @author WVN
            std::string NumberToHexValue(double d)
            {
                if (d < 0) return "00";
                if (d > 1) return "ff";
                std::string output = "";
                //This is silly but I am tired
                std::string hexVals = "0123456789abcdef";
                int v = d*0x000000ff;
                output = output + hexVals[v&0x0000000f];
                output = output + hexVals[(v>>4)&0x0000000f];
                return output;
            }
    };
}

#endif