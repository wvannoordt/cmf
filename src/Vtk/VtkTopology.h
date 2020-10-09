#ifndef VTK_TOPOLOGY_H
#define VTK_TOPOLOGY_H

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "VtkElement.h"
#include "VtkAttributeCollection.h"

namespace cmf
{
    namespace VtkTopologyType
    {
        enum VtkTopologyType
        {
            structuredPoints,
            structuredGrid,
            unstructuredGrid,
            polydata,
            rectilinearGrid,
            field
        };
    }

    inline static std::string VtkTopologyTypeStr(VtkTopologyType::VtkTopologyType ttype)
    {
        switch (ttype)
        {
            case VtkTopologyType::structuredPoints: return "STRUCTURED_POINTS";
            case VtkTopologyType::structuredGrid:   return "STRUCTURED_GRID";
            case VtkTopologyType::unstructuredGrid: return "UNSTRUCTURED_GRID";
            case VtkTopologyType::polydata:         return "POLYDATA";
            case VtkTopologyType::rectilinearGrid:  return "RECTILINEAR_GRID";
            case VtkTopologyType::field:            return "FIELD";
        }
        return "NONE";
    }

    /// @brief Class representing various topology types supported in VTK file format
    /// \see VtkTopologyType
    /// @author WVN
    class VtkTopology : public VtkElement, public VtkHasAttributableCollection
    {
        public:
            /// @brief Constructor for the VtkTopology object
            /// @param topologyType_in Represents the VtkTopologyType
            /// @author WVN
            VtkTopology(VtkTopologyType::VtkTopologyType topologyType_in);
            
            /// @brief Empty constructor for the VtkTopology object
            /// @author WVN
            VtkTopology(void);
            
            /// @brief Destructor for the VtkTopology object
            /// @author WVN
            ~VtkTopology(void);
            
            /// @brief Writes topology to file.
            /// @param myfile File stream to be written to 
            /// @author WVN
            void WriteToFile(std::ofstream & myfile);
            
            /// @brief Reads topology from file.
            /// @param myfile File stream to be read from
            /// @author WVN
            void ReadFromFile(std::ofstream & myfile);
            
            /// @brief Sets up required attributes for the given topology type.
            /// @author WVN
            void InitTopology(void);
            
        private:
            /// @brief The topology type
            VtkTopologyType::VtkTopologyType topologyType;
            
            /// @brief Set to true when empty constructor is called
            bool uninitialized;
    };
}

#endif
