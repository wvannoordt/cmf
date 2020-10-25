#ifndef CMF_GEOMETRY_CONVERTER
#define CMF_GEOMETRY_CONVERTER
#include "SurfaceTriangulation.h"
#include "CmfScreen.h"
#include <string>
namespace cmf
{
    class SurfaceTriangulation;
    /// @brief A class for generically converting triangulation data to the CMF standard format
    /// @author WVN
    class ICmfGeometryConverter
    {
        public:
            
            /// @brief Empty constructor
            /// @author WVN
            ICmfGeometryConverter(void){}
            
            /// @brief Empty destructor
            /// @author WVN
            ~ICmfGeometryConverter(void){}
            
            /// @brief Converts internal geometry data to the CMF standard format
            /// @param target The triangulation to define
            /// @author WVN
            virtual void ConvertGeometry(SurfaceTriangulation* target){}
            
            /// @brief Writes a point cloud CSV file
            /// @param filename The file to write to
            /// @author WVN
            virtual void SavePointCloud(std::string filename){}
    };
}

#endif