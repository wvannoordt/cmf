#include "VtkTopology.h"
namespace Anaptric
{
    VtkTopology::VtkTopology(VtkTopologyType::VtkTopologyType topologyType_in)
    {
        elementType = "topology";
        topologyType = topologyType_in;
    }
    
    VtkTopology::VtkTopology(void)
    {
        elementType = "topology";
        topologyType = VtkTopologyType::polydata;
    }
    
    VtkTopology::~VtkTopology(void)
    {
        
    }
    
    void VtkTopology::WriteToFile(std::ofstream & myfile)
    {
        myfile << "DATASET " << VtkTopologyTypeStr(topologyType) << std::endl;
    }
    
    void VtkTopology::ReadFromFile(std::ofstream & myfile)
    {
        
    }
}