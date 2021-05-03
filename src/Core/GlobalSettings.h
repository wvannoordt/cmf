#ifndef CMF_GLOBAL_SETTINGS_H
#define CMF_GLOBAL_SETTINGS_H
#include "ICmfInputObject.h"
#include "PTL.h"
namespace cmf
{   
    /// @brief A struct for global settings for an instance of CMF
    /// @author WVN
    struct GlobalSettings : public ICmfInputObject
    {
        /// @brief If set to false the absolutely no output will come from CmfScreen/WriteLine (used for parallel programming to prevent too much output)
        bool globalOutputEnabledHere;

        /// @brief Level of debugging output. Higher level means more output, lower means less. 1 to 8
        int debugLevel;

        /// @brief If set to true, will output file and line numbers for every call to WriteLine
        bool trackOutputOrigins;
        
        /// @brief If set to true, will enable the emulated stack
        bool allowStackAllocation;
        
        /// @@brief indicates whether or not the GlobalSettings object was parsed from a file
        bool hasBeenParsedFromFile;
        
        /// @@brief indicates whether or not outputs can be colored, not recommended if outputting stream to file.
        bool colorOutput;

        /// @brief Constructor for the GlobalSettings object.
        /// @param title_in title of the mesh
        /// @param inputTree PropertryTree to be read from
        /// @author WVN
        GlobalSettings(PTL::PropertySection& inputTree) : ICmfInputObject(inputTree)
        {
            globalOutputEnabledHere = true;
            Define(*objectInput);
            Parse();
            hasBeenParsedFromFile = true;
        }

        /// @brief Default constructor for the GlobalSettings object.
        /// @author WVN
        GlobalSettings(void)
        {
            objectInput = NULL;
            globalOutputEnabledHere = true;
            debugLevel = 1;
            trackOutputOrigins = false;
            hasBeenParsedFromFile = false;
        }

        /// @brief Defines the input variables
        /// @author WVN
        void Define(PTL::PropertySection& input)
        {
            input["Display"]["debugLevel"].MapTo(&debugLevel) = new PTL::PTLInteger(1, "Level of debug output written to screen");
            input["Display"]["trackOutputOrigins"].MapTo(&trackOutputOrigins) = new PTL::PTLBoolean(false, "Writes file and line number to screen along with output");
            input["Display"]["colorOutput"].MapTo(&colorOutput) = new PTL::PTLBoolean(false, "Enables color-coded terminal output. Not recommended when running on non-ANSI color-code supporting terminals or HPC systems");
            
            input["Memory"]["allowStackAllocation"].MapTo(&allowStackAllocation) = new PTL::PTLBoolean(false, "If set to true, will enable the emulated stack (use \"false\" for better debugging)");
        }
    };
}
#endif
