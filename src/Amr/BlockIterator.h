#ifndef BLOCK_ITERATOR_H
#define BLOCK_ITERATOR_H
#include <vector>
#include <iostream>
#include <ostream>
#include "AmrFcnTypes.h"
#include "CmfOutputStream.h"
#include "IBlockIterable.h"
#include "CmfMPI.h"
namespace cmf
{
    class RefinementBlock;
    
    namespace IterableMode
    {
        enum IterableMode
        {
            parallel,
            serial
        };
    }
    
    /// @brief A class that is used to iterate over the nodes in a block.
    /// @author WVN
    class BlockIterator
    {
        public:
            /// @brief Constructor for the iterator
            /// @param hostBlock_in The block to iterate over the nodes of
            /// @author WVN
            BlockIterator(IBlockIterable* hostBlock_in);
            
            /// @brief Constructor for the iterator
            /// @param hostBlock_in The block to iterate over the nodes of
            /// @param filter_in A function pointer that, if returns false on a given node, causes the iterator to skip that node
            /// @author WVN
            BlockIterator(IBlockIterable* hostBlock_in, NodeFilter_t filter_in);
            
            /// @brief Constructor for the iterator
            /// @param hostBlock_in The block to iterate over the nodes of
            /// @param filter_in A function pointer that, if returns false on a given node, causes the iterator to skip that node
            /// @param mode_in Indicates whether or not this iterator should loop over nodes that are not contained by the host block's parallel partition \see IterableMode
            /// @author WVN
            BlockIterator(IBlockIterable* hostBlock_in, NodeFilter_t filter_in, IterableMode::IterableMode mode_in);
            
            /// @brief Constructor for the iterator
            /// @param hostBlock_in The block to iterate over the nodes of
            /// @param mode_in Indicates whether or not this iterator should loop over nodes that are not contained by the host block's parallel partition \see IterableMode
            /// @author WVN
            BlockIterator(IBlockIterable* hostBlock_in, IterableMode::IterableMode mode_in);
            
            /// @brief Destructor for the iterator
            /// @author WVN
            ~BlockIterator(void);
            
            /// @brief Determines if the iterator has reached the end of the list.
            /// @author WVN
            bool HasNext(void);
            
            /// @brief Increments the current iterator (postfix).
            /// @param dummy Unused
            /// @author WVN
            BlockIterator   operator++(int dummy);
            
            /// @brief Increments the current iterator (prefix).
            /// @author WVN
            BlockIterator & operator++(void);
            
            /// @brief Returns the current relevant node.
            /// @author WVN
            RefinementTreeNode* Node(void);
            
            /// @brief Returns the number of current relevant nodes.
            /// @author WVN
            size_t Size(void);
            
            /// @brief Prints the current index as an integer when streamed.
            /// @author WVN
            friend std::ostream & operator << (std::ostream &out, const BlockIterator &c) {out << c.index; return out;}
            
            /// @brief Prints the current index as an integer when streamed.
            /// @author WVN
            friend CmfOutputStream & operator << (CmfOutputStream &out, const BlockIterator &c) {out << c.index; return out;}
        private:
            /// @brief Moves to the first node that satisfies the filter
            /// @author WVN
            void SeekFirst(void);
            
            /// @brief Acts as hidden constructor.
            /// @param hostBlock_in The block to iterate over the nodes of
            /// @param filter_in A function pointer that, if returns false on a given node, causes the iterator to skip that node
            /// @param mode_in Indicates whether or not this iterator should loop over nodes that are not contained by the host block's parallel partition \see IterableMode
            /// @author WVN
            void Build(IBlockIterable* hostBlock_in, NodeFilter_t filter_in, IterableMode::IterableMode mode_in);
            
            /// @brief The block being iterated over
            IBlockIterable* hostBlock;
            
            /// @brief The current index with respect to allNodes
            size_t index;
            
            /// @brief The function pointer that, if returns false on a given node, causes the iterator to skip that node
            NodeFilter_t filter;
            
            /// @brief The list of all relevant nodes
            std::vector<RefinementTreeNode*>* allNodes;
            
            /// @brief Either serial or parallel. If parallel, this iteraator only loops over nodes on the host block's parallel partition
            IterableMode::IterableMode parallelMode;
            
            /// @brief Indicates whether or not the iterator has reached the end of the underlyin vector
            bool isAtEnd;
    };
}

#endif
