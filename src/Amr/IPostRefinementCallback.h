#ifndef CMF_IPOST_REFINEMENT_CALLBACK_H
#define CMF_IPOST_REFINEMENT_CALLBACK_H
#include "RefinementTreeNode.h"
#include "RefinementBlock.h"
#include <vector>
namespace cmf
{
    /// @brief Interface for an object that implements a post-mesh-refinement callback
    /// @author WVN
    class IPostRefinementCallback
    {
        public:
            /// @brief Empty constructor
            /// @author WVN
            IPostRefinementCallback(void){}
            
            /// @brief Empty destructor
            /// @author WVN
            ~IPostRefinementCallback(void){}
            
            /// @brief The callback function for new nodes
            /// @param newNodes The newly refined nodes to be handled
            /// @author WVN
            virtual void OnPostRefinementCallback(std::vector<RefinementTreeNode*>& newNodes)=0;
            
            /// @brief Adds the current object as a post-refinement callback object to the provided AMR block
            /// @param blocks The blocks to register this object to
            /// @author WVN
            void RegisterToBlocks(RefinementBlock* blocks)
            {
                callbackOrder = blocks->AddPostRefinementCallbackObject(this);
            }
        
        protected:
            
            /// @brief The callback order of the current object
            int callbackOrder;
    };
}

#endif