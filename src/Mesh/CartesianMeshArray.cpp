#include "CartesianMeshArray.h"
#include "DebugTools.hx"
#include "CartesianMeshArrayHandler.h"
namespace cmf
{
    CartesianMeshArray::CartesianMeshArray(ArrayInfo info, CartesianMeshArrayHandler* handler_in) : ICmfMeshArray(info)
    {
        handler = handler_in;
    }
    
    CartesianMeshArray::~CartesianMeshArray(void)
    {
        
    }
    
    void CartesianMeshArray::Destroy(void)
    {
        
    }
}