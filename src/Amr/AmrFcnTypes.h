#ifndef AMR_FCN_TYPES_H
#define AMR_FCN_TYPES_H
namespace cmf
{
    class RefinementTreeNode
    {
        public:
            bool IsTerminal(void);
    };
    class BlockIterator;
    class TikzObject;
    typedef bool(*NodeFilter_t)(RefinementTreeNode*);
    typedef void(*DebugTikzDraw_t)(TikzObject*, RefinementTreeNode*);

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
