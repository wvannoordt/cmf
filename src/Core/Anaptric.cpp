#include "PropTreeLib.h"
#include "Anaptric.h"
#include <string>
namespace Anaptric
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
