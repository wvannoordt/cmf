#include "ICmfMeshArray.h"
#include "CmfScreen.h"
namespace cmf
{
    ICmfMeshArray::ICmfMeshArray(ArrayInfo info)
    {
        variableName = info.name;
        if (info.rank > MAX_RANK) CmfError("Rank of variable \"" + variableName + "\" exceeds MAX_RANK (" + std::to_string(MAX_RANK) + "): recompile with greater limit.");
        for (int i = 0; i < info.rank; i++)
        {
            dims[i] = info.dimensions[i];
        }
        elementSize = info.elementSize;
    }
    
    ICmfMeshArrayHandler* ICmfMeshArray::GetHandler(void)
    {
        return NULL;
    }
    
    void ICmfMeshArray::Destroy(void)
    {
        
    }
    
    ICmfMeshArray::~ICmfMeshArray(void)
    {
        
    }
}