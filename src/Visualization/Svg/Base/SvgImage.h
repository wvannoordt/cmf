#ifndef CMF_SVG_IMAGE_H
#define CMF_SVG_IMAGE_H
// Useful link:
// https://www.w3.org/TR/SVG/eltindex.html
#include <map>
#include "SvgElement.h"
#include "SvgElementGroup.h"
#include "SvgElementHandler.h"
#include "SvgColor.h"
namespace cmf
{
    /// @brief Represents a coordinate transformation to be applied to the screen coordinates,
    /// i.e. x_out = M*x_in + b, where M = [m11 m12;m21 m22] and b = [b1;b2]
	/// @author WVN
    struct ImageTransformation
    {
        double m11, m12, m21, m22, b1, b2;
    };
    /// @brief Class that represents an SVG image
	/// @author WVN
    class SvgImage : public SvgElementHandler
    {
        public:
            /// @brief Default constructor
        	/// @author WVN
            SvgImage(void);
            
            /// @brief Constructor
            /// @param xmin left side of the image (coordinate)
            /// @param xmax right side of the image (coordinate)
            /// @param ymin lower side of the image (coordinate)
            /// @param ymax upper side of the image (coordinate)
        	/// @author WVN
            SvgImage(double xmin, double xmax, double ymin, double ymax);
            
            /// @brief Sets the image bounding box
            /// @param xmin left side of the image (coordinate)
            /// @param xmax right side of the image (coordinate)
            /// @param ymin lower side of the image (coordinate)
            /// @param ymax upper side of the image (coordinate)
            /// @author WVN
            void SetBounds(double xmin, double xmax, double ymin, double ymax);
            
            /// @brief Destructor
        	/// @author WVN
            ~SvgImage(void);
            
            /// @brief Creates a new group
            /// @param name The name of the group to create
        	/// @author WVN
            SvgElementGroup* CreateGroup(std::string name);
            
            /// @brief Returns an existing group
            /// @param name The name of the group to create
        	/// @author WVN
            SvgElementGroup* GetGroup(std::string name);
            
            /// @brief Returns a group by name and creates it if it does not already exist
            /// @param name The name of the group to return
        	/// @author WVN
            SvgElementGroup& operator [] (std::string name);
            
            /// @brief Writes the image to a file
            /// @param filename The name of the file to write
        	/// @author WVN
            void Write(std::string filename);
            
            /// @brief Maps a point from geometric coordinates to screen coordinates
            /// @param xin X-coordinate of the input point
            /// @param yin Y-coordinate of the input point
            /// @param xout X-coordinate of the output point
            /// @param yout Y-coordinate of the output point
        	/// @author WVN
            double MapPoint(double xin, double yin, double* xout, double* yout);
            
            /// @brief Sets the fill color for the background of the image
            /// @param color The color to fill
        	/// @author WVN
            void SetFillColor(std::string color);
            
            /// @brief Checks if a given name is reserved for internal use
            /// @param name The name to check
        	/// @author WVN
            bool NameIsReserved(std::string name);
            
            /// @brief Returns the list of groups
            /// @author WVN
            std::vector<SvgElementGroup*>& GetGroups(void);
            
            /// @brief Returns the list of group names
            /// @author WVN
            std::vector<std::string>& GetGroupNames(void);
            
            /// @brief Returns the table of group layer IDs
            /// @author WVN
            std::map<std::string, int>& GetGroupTable(void);
            
        private:
            
            /// @brief Creates a new group
            /// @param name The name of the group to create
            /// @param ignoreReserved If true, ignores if a name is reserved
        	/// @author WVN
            SvgElementGroup* CreateGroup(std::string name, bool ignoreReserved);
            
            /// @brief Builder function
            /// @param xmin left side of the image (coordinate)
            /// @param xmax right side of the image (coordinate)
            /// @param ymin lower side of the image (coordinate)
            /// @param ymax upper side of the image (coordinate)
            /// @author WVN
            void Build(double xmin, double xmax, double ymin, double ymax);
            
            /// @brief Returns true if the image has a group with the given name
            /// @param name The name to check
            /// @author WVN
            bool HasGroup(std::string name);
            
            /// @brief The list of element groups (each containing elements) in the image
            std::map<std::string, int> elementLocations;
            
            /// @brief The list of element group names in the image
            std::vector<std::string> elementNames;
            
            /// @brief The list of element groups in the image
            std::vector<SvgElementGroup*> elementGroups;
            
            /// @brief The default (global) group
            SvgElementGroup* defaultGroup;
            
            /// @brief The image bounds (coordinates): (left, right, lower, upper)
            double bounds[4];
            
            /// @brief A list of image transformations to be applied
            std::vector<ImageTransformation> transforms;
            
            /// @brief Indicates whether or not the image will have a filled background
            bool hasFillColor;
            
            /// @brief A color to fill the backgroud with, if enabled
            SvgColor fillColor;
    };
}

#endif