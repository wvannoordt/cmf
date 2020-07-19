#include <string>
#include "PropTreeLib.h"
#include "RefinementTree.h"

namespace gTree
{
    RefinementTree(std::string title)
    {
        localInput.SetAsSubtree(mainInput[title]);
    }

    ~RefinementTree(void)
    {
        
    }
}

#endif
