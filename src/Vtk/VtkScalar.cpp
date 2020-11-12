#include "VtkScalar.h"

namespace cmf
{
    VtkScalar::VtkScalar(void)
    {
        VtkAttributable* scalarData = collection.AddAttributable("SCALARS", VtkAttributableType::doubleType);
        scalarData->AddRequiredAttribute("name", VtkAttributableType::stringType);
        scalarData->AddRequiredAttribute("numPoints", VtkAttributableType::longType);
        scalarData->AddRequiredAttribute("bufferCount", VtkAttributableType::longType);
        scalarData->AddRequiredAttribute("stride", VtkAttributableType::intType);
        scalarData->SetFormat("SCALARS ${name} double\nLOOKUP_TABLE default");
    }
    
    VtkScalar::~VtkScalar(void)
    {
        
    }
    
    void VtkScalar::WriteToFile(std::ofstream & myfile)
    {
        collection.WriteAll(myfile);
    }
    
    void VtkScalar::ReadFromFile(std::ofstream & myfile)
    {
        
    }
}