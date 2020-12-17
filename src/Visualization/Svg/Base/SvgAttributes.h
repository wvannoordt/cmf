#ifndef CMF_SVG_ATTR_H
#define CMF_SVG_ATTR_H
#include <string>
#include <map>
#include "CmfError.h"
namespace cmf
{
    /// @brief Represents a list of SVG attributes
	/// @author WVN
    class SvgAttributes
    {
        public:
            /// @brief Empty constructor
        	/// @author WVN
            SvgAttributes(void){}
            /// @brief Empty destructor
        	/// @author WVN
            ~SvgAttributes(void){}
            
            /// @brief Adds a string attribute
            /// @param id The name of the attribute
            /// @param val The value of the attribute
        	/// @author WVN
            void Add(std::string id, std::string val)
            {
                names.push_back(id);
                values.push_back(val);
                attributes.insert({id, values.size()-1});
            }
            
            /// @brief Adds a double attribute
            /// @param id The name of the attribute
            /// @param val The value of the attribute
        	/// @author WVN
            void Add(std::string id, double val)
            {
                Add(id, std::to_string(val));
            }
            
            /// @brief Adds an integer attribute
            /// @param id The name of the attribute
            /// @param val The value of the attribute
        	/// @author WVN
            void Add(std::string id, int val)
            {
                Add(id, std::to_string(val));
            }
            
            /// @brief Returns the XML-formatted string of all attributes
        	/// @author WVN
            std::string GetString(void)
            {
                std::string output = "";
                for (int i = 0; i < names.size(); i++)
                {
                    if (i > 0) output = output + " ";
                    output = output + names[i] + "=\"" + values[i] + "\"";
                }
                return output;
            }
        private:
            /// @brief List of attribute locations
            std::map<std::string, int> attributes;
            
            /// @brief List of attribute names
            std::vector<std::string> names;
            
            /// @brief List of attribute values
            std::vector<std::string> values;
    };
}

#endif