#ifndef CMF_ICMFGEOMETRY_H
#define CMF_ICMFGEOMETRY_H
#include "Config.h"
#include "CmfError.h"
#include <cmath>
namespace cmf
{
    /// @brief A general class that is used to represent arbitrary geometries
    /// @author WVN
    class ICmfGeometry
    {
        public:
            /// @brief Constructor for the base class
            /// @author WVN
            ICmfGeometry(void) {geometryType = "NONE";}
            
            /// @brief Destructor for the base class
            /// @author WVN
            ~ICmfGeometry(void) {}
            
            /// @brief Determines whether the provided box intersects the boundary of the geometry
            /// @param bounds The bounds of the box to check (xmin, xmax, ymin, ymax, [zmin, zmax])
            /// @author WVN
            virtual bool BoxIntersectsBoundary(double* bounds){};
            
        protected:
            
            /// @brief A name for the current type of geometry
            std::string geometryType;
    };
}

#endif