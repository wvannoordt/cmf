#ifndef ICMF_MESH_BUFFER_H
#define ICMF_MESH_BUFFER_H
#include "ParallelFile.h"
namespace cmf
{
    /// @brief A base class for a mesh data buffer, which handles the memory allocated to a mesh array
    /// @author WVN
    class ICmfMeshBuffer
    {
        /// @brief Writes the binary mesh data to the provided file
        /// @param file The file to write to
        /// @author WVN
        virtual void WriteMeshDataToFile(ParallelFile& file)=0;
        
        /// @brief Reads the binary mesh data from the provided file
        /// @param file The file to read from
        /// @author WVN
        virtual void ReadMeshDataFromFile(ParallelFile& file)=0;
    };
}

#endif