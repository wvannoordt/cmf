#ifndef ICMF_INPUT_OBJECT_H
#define ICMF_INPUT_OBJECT_H
#include <string>
#include "PropTreeLib.h"
#include "CmfError.h"
namespace cmf
{
    /// @brief Defines a general input object to be read from an input file
    /// @author WVN
    struct ICmfInputObject
    {
        /// @brief The title of this object to be read from the input file as a section
        std::string title;
        
        /// @brief A PropertyTree local to this object
        PropTreeLib::PropertyTree objectInput;
        
        /// @brief Parses the options in the input file
        /// @author WVN
        void Parse(void){objectInput.StrictParse();}
        
        /// @brief Input-less constructor provided for manual construction
        /// @param title_in The title of this object, to be read from the input tree
        /// @author WVN
        ICmfInputObject(std::string title_in) {title = title_in;}
        
        /// @brief Constructor for a general inpt object
        /// @param title_in The title of this object, to be read from the input tree
        /// @param inputTree A PropertyTree to be read from. (See <a href="https://github.com/wvannoordt/PropTreeLib">PropTreeLib</a> )
        /// @author WVN
        ICmfInputObject(std::string title_in, PropTreeLib::PropertyTree & inputTree) {title = title_in; objectInput.SetAsSubtree(inputTree[title]);}
        
        /// @brief Defines a general input object to be read from an input file
        /// @param inputTree A PropertyTree to be read from. (See <a href="https://github.com/wvannoordt/PropTreeLib">PropTreeLib</a> )
        /// @author WVN
        virtual void Define(void) {}
    };
}

#endif