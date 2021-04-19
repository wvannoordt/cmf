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
}
