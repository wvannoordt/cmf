#ifndef CMF_CARTESIAN_MESH_ARRAY_PARALLEL_VTK_WRITER_H
#define CMF_CARTESIAN_MESH_ARRAY_PARALLEL_VTK_WRITER_H
#include "CartesianMeshArray.h"
#include "StringUtils.h"
#include "Vec.h"
namespace cmf
{
    /// @brief Object for writing a cartesian mesh array to a parallel vtk format
    /// @author WVN
    class CartesianMeshArrayParallelVtkWriter
    {
        public:
            /// @brief Constructor
            /// @param directory_in The directory to output the file to
            /// @param fileTitle_in The base name of the file, with no file extension
            /// @author WVN
            CartesianMeshArrayParallelVtkWriter(std::string directory_in, std::string fileTitle_in);
            
            /// @brief Destructor
            /// @author WVN
            ~CartesianMeshArrayParallelVtkWriter(void);
            
            /// @brief Constructor
            /// @param array The array to output
            /// @author WVN
            void Export(CartesianMeshArray& array);
        
        private:
            
            /// @brief Returns the name of the meta-file
            /// @param array The array being output
            /// @author WVN
            std::string GetMetaFileName(CartesianMeshArray& array);
            
            /// @brief Returns the template name of a block file relative to the metafile
            /// @param array The array being output
            /// @author WVN
            std::string GetBlockFileTemplateRelativeToMetaFile(CartesianMeshArray& array);
            
            
            /// @brief Returns the template name of a block file
            /// @param array The array being output
            /// @author WVN
            std::string GetBlockFileTemplate(CartesianMeshArray& array);
            
            /// @brief Returns the largest block dimensions, including exchange cells
            /// @param array The array being output
            /// @author WVN
            Vec3<int> GetLargestBlockDims(CartesianMeshArray& array);
            
            /// @brief directory The directory to output the file to
            std::string outputDirectory;
            
            /// @brief fileTitle The base name of the file, with no file extension
            std::string fileTitle;
    };
}

#endif