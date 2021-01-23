#ifndef CMF_SVG_LINE_H
#define CMF_SVG_LINE_H
#include "SvgElement.h"
#include "SvgHasStroke.h"
#include "SvgNode.h"
namespace cmf
{
    /// @brief Class that represents a line on an svg image
	/// @author WVN
    class SvgLine : public SvgElement, public SvgHasStroke
    {
        public:
            /// @brief Constructor
            /// @param startNode_in Coordinates of the initial point
            /// @param endNode_in Coordinates of the final point
        	/// @author WVN
            SvgLine(SvgNode* startNode_in, SvgNode* endNode_in, SvgImage* host);
            
            /// @brief Creates the relevant attributes for file output
        	/// @author WVN
            void CreateAttributes(void);
            
            /// @brief Destructor
        	/// @author WVN
            ~SvgLine(void);
            
        private:
            
            /// @brief The node at the beginning of the line
            SvgNode* startNode;
            
            /// @brief The node at the end of the line
            SvgNode* endNode;
    };
}

#endif