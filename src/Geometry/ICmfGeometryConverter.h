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
            
            /// @brief 
            /// @param filename_in The file name to convert from/to
            /// @param target_in The surface triangulation object to convert to/from
            /// @author WVN
            ICmfGeometryConverter(std::string filename_in, SurfaceTriangulation* target_in)
            {
                filename = filename_in;
                target = target_in;
            }
            
            /// @brief Empty destructor
            /// @author WVN
            ~ICmfGeometryConverter(void){}
            
            /// @brief Converts internal geometry data to the CMF standard format
            /// @author WVN
            virtual void ConvertGeometry(void){}
            
            /// @brief Writes a point cloud CSV file (for debugging)
            /// @param filename The file to write to
            /// @author WVN
            virtual void SavePointCloud(std::string filename){}
            
            /// @brief Saves the native geometry as the corresponding file type
            /// @author WVN
            virtual void SaveGeometry(void){}
        
        protected:
            ///@brief The filename for this geometry converter (to either be written to or read from)
            std::string filename;
            
            /// @brief The surface triangulation to read to or write to
            SurfaceTriangulation* target;
    };
}

#endif