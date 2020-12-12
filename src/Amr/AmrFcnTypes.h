#ifndef AMR_FCN_TYPES_H
#define AMR_FCN_TYPES_H
namespace cmf
{
    class RefinementTreeNode;
    class BlockIterator;
    class TikzObject;
    typedef bool(*NodeFilter_t)(RefinementTreeNode*);
    typedef void(*DebugTikzDraw_t)(TikzObject*, RefinementTreeNode*);
}
#endif