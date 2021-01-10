#include "PTL.h"
#include "cmf.h"
#include <string>
#include "CmfCuda.h"
namespace cmf
{
    PTL::PropertyTree mainInput;
    GlobalSettings globalSettings;
    void Initialize(void)
    {

    }

    void ReadInput(std::string filename)
    {
        mainInput.ReadInputFileToTreeData(filename);
    }

    void Finalize(void)
    {

    }
    bool HasGpuSupport(void)
    {
        return CUDA_ENABLE?true:false;
    }
    
    bool IsParallel(void)
    {
#if(CMF_PARALLEL)
        return true;
#else
        return false;
#endif
    }

    int GetDim(void)
    {
        //To test if a 2D code calls a 3D api etc.
        return CMF_DIM;
    }
}
