#include "PropTreeLib.h"
#include <string>
namespace gTree
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
}
