#ifndef CMF_BLOCK_ITERABLE_H
#define CMF_BLOCK_ITERABLE_H
#include <vector>
namespace cmf
{
    class RefinementTreeNode;
    class RefinementBlock;
    class IBlockIterable
    {
        /// @brief Defines an object that can have its blocks iterated over
        /// @author WVN
        public:
            /// @brief Empty constructor
            /// @author WVN
            IBlockIterable(void){}
            
            /// @brief Empty destructor
            /// @author WVN
            ~IBlockIterable(void){}
            
            /// @brief Returns the total number of nodes that are contained within the iterable object
            /// @author WVN
            virtual size_t Size(void) {return 0;}
            
            /// @author WVN
            /// @brief Gets the list of blocks to be iterated over
            virtual std::vector<RefinementTreeNode*>* GetAllNodes(void){return NULL;}
            
            /// @author WVN
            /// @brief Returns true if the iterable object has this node in its parallel partition
            /// @param node The node to check
            virtual bool ParallelPartitionContainsNode(RefinementTreeNode* node) {return true;}
            
            /// @author WVN
            /// @brief Returns the RefinementBlock object containing the nodes iterated over
            /// refinements, such as variable interpolation and buffer expansion
            virtual RefinementBlock* GetRefinementBlockObject(void)=0; //set to null here because every implementation of this interface should define this explicitly!!
    };
}

#endif