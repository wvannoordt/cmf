#ifndef CMF_SERIAL_CARTESIAN_VTK_H
#define CMF_SERIAL_CARTESIAN_VTK_H
#include "AmrFcnTypes.h"
#include "VtkFile.h"
#include "CartesianMesh.h"
#include "RefinementTreeNode.h"
#include <string>
#include <vector>
namespace cmf
{
    /// @brief A class for outputting an unstructured VTK file containing data. NOTE: this is generally very slow and should only be used for debugging
    /// @author WVN
    class SerialCartesianVtk
    {
        public:
            /// @brief Constructor
            /// @param mesh_in The mesh to output
            /// @param filename_in The name of the file to be output
            /// @param filter_in The block filter to apply to the mesh
            /// @author WVN
            SerialCartesianVtk(CartesianMesh& mesh_in, std::string filename_in, NodeFilter_t filter_in);
            
            /// @brief Constructor
            /// @param mesh_in The mesh to output
            /// @param filename_in The name of the file to be output
            /// @author WVN
            SerialCartesianVtk(CartesianMesh& mesh_in, std::string filename_in);
            
            /// @brief Destructor
            /// @author WVN
            ~SerialCartesianVtk(void);
            
            /// @brief Writes the vtkFile
            /// @author WVN
            void Write(void);
            
            /// @brief Stream operator allowing mesh arrays to be placed into VtkData arrays in the current object
            /// @param variable The variable to place on the Vtk mesh
            /// @author WVN
            SerialCartesianVtk& operator << (CartesianMeshArray& variable);
        
        private:
            
            /// @brief Builder function, used essentailly as a constructor
            /// @param mesh_in The mesh to output
            /// @param filename_in The name of the file to be output
            /// @param filter_in The block filter to apply to the mesh
            /// @author WVN
            void Build(CartesianMesh& mesh_in, std::string filename, NodeFilter_t filter_in);
            
            /// @brief Fills the vtk object with mesh data and builds the relevant block list
            /// @author WVN
            void ProcessBlocks(void);
            
            /// @brief Release the resources used by this object
            /// @author WVN
            void Destroy(void);
            
            /// @brief VTK file object
            VtkFile* output;
            
            /// @brief File name to write to
            std::string filename;
            
            /// @brief List of relevant output blocks
            std::vector<RefinementTreeNode*> blocks;
            
            /// @brief A pointer to the underlying mesh
            CartesianMesh* mesh;
            
            /// @brief A filter defining the domain of the output
            NodeFilter_t filter;
            
            /// @brief Indicates whether or not the VtkFile must be deleted
            bool requireDelete;
            
            /// @brief Total number of points on the mesh
            size_t totalNumPointsOnMesh;
    };
}

#endif
