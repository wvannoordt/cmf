#include "CartesianMeshExchangeHandler.h"
#include "CartesianMesh.h"
namespace cmf
{
    CartesianMeshExchangeHandler::CartesianMeshExchangeHandler(CartesianMesh* mesh_in, CartesianMeshExchangeInfo& inputInfo)
    {
        mesh = mesh_in;
        interpolationOrder = inputInfo.interpolationOrder;
        WriteLine(2, "Create exchange pattern on mesh \"" + mesh->GetTitle() + "\"");
        exchangeDim = inputInfo.exchangeDim;
        
    }
    CartesianMeshExchangeHandler::~CartesianMeshExchangeHandler(void)
    {
        
    }
}