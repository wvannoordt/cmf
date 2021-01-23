#ifndef CMF_SVG_NODE_H
#define CMF_SVG_NODE_H

namespace cmf
{
    /// @brief Class that represents a referenceable point on an svg image
	/// @author WVN
    struct SvgNode
    {
        /// @brief Constructor
        /// @param x_in x-coordinate of the node
        /// @param y_in y-coordinate of the node
    	/// @author WVN
        SvgNode(double x_in, double y_in) {x=x_in; y=y_in;}
        
        /// @brief x-coordinate of the node
        double x;
        
        /// @brief y-coordinate of the node
        double y;
    };
}

#endif