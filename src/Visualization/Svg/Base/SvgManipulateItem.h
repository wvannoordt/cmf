#ifndef CMF_SVG_MANIP_ITEM_H
#define CMF_SVG_MANIP_ITEM_H

namespace cmf
{
    class SvgImage;
    class SvgElementGroup;
    /// @brief Class that represents an object that can be manipulated in an SVG, i.e. a group/layer and an SvgElement
	/// @author WVN
    class SvgManipulateItem
    {
        public:
            /// @brief Constructor
        	/// @author WVN
            SvgManipulateItem(void);
            
            /// @brief Destructor
        	/// @author WVN
            ~SvgManipulateItem(void);
        
            /// @brief Sets the visibility of this item
            /// @param val The value to set
            /// @author WVN
            bool SetVisibility(bool val);
            
            /// @brief Returns the visibility of this item
            /// @author WVN
            bool IsVisible(void);
            
            /// @brief Returns the relevant image
            /// @author WVN
            virtual SvgImage* GetImage(void)=0;
            
            /// @brief Returns the relevant group
            /// @author WVN
            virtual SvgElementGroup* GetGroup(void)=0;
            
            /// @brief Sets the position of this item
            /// @param pos The value to set
            /// @author WVN
            void SetPosition(int pos) {containerPosition = pos;}
                    
        protected:
            
            /// @brief Indicates whether the item is visible
            bool visible;
            
            /// @brief Indicates the position of the current manipulateable element within its container
            int containerPosition;
    };
}

#endif