#ifndef CMF_BLOCKFILTERS_H
#define CMF_BLOCKFILTERS_H
#include "AmrFcnTypes.h"
namespace cmf
{
    namespace BlockFilters
    {
        static inline bool everyBlockFilter(RefinementTreeNode* node)
        {
            return true;
        }
        static NodeFilter_t Every(everyBlockFilter);

        static inline bool terminalBlockFilter(RefinementTreeNode* node)
        {
            return node->IsTerminal();
        }
        static NodeFilter_t Terminal(terminalBlockFilter);
    }
}
#endif