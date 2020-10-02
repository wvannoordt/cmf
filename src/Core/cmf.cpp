#include "PropTreeLib.h"
#include "cmf.h"
#include <string>
namespace cmf
{
    PropTreeLib::PropertyTree mainInput;
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

    int GetDim(void)
    {
        //To test if a 2D code calls a 3D api etc.
        return CMF_DIM;
    }
}
