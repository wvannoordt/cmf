#ifndef CMF_GLOBAL_SETTINGS_H
#define CMF_GLOBAL_SETTINGS_H
#include "ICmfInputObject.h"
#include "PropTreeLib.h"
namespace cmf
{   
    /// @brief A struct for global settings for an instance of CMF
    /// @author WVN
    struct GlobalSettings : ICmfInputObject
    {
        /// @brief If set to false the absolutely no output will come from CmfScreen/WriteLine (used for parallel programming to prevent too much output)
        bool globalOutputEnabledHere;

        /// @brief Level of debugging output. Higher level means more output, lower means less. 1 to 8
        int debugLevel;

        /// @brief If set to true, will output file and line numbers for every call to WriteLine
        bool trackOutputOrigins;
        
        /// @brief If set to true, will enable the emulated stack
        bool allowStackAllocation;

        /// @brief Constructor for the GlobalSettings object.
        /// @param title_in title of the mesh
        /// @param inputTree PropertryTree to be read from
        /// @author WVN
        GlobalSettings(PropTreeLib::PropertySection& inputTree) : ICmfInputObject(inputTree)
        {
            globalOutputEnabledHere = true;
            Define(*objectInput);
            Parse();
        }

        /// @brief Default constructor for the GlobalSettings object.
        /// @author WVN
        GlobalSettings(void)
        {
            objectInput = NULL;
            globalOutputEnabledHere = true;
            debugLevel = 1;
            trackOutputOrigins = false;
        }

        /// @brief Defines the input variables
        /// @author WVN
        void Define(PropTreeLib::PropertySection& input)
        {
            input["Display"]["debugLevel"].MapTo(&debugLevel) = new PropTreeLib::Variables::PTLInteger(1, "Level of debug output written to screen");
            input["Display"]["trackOutputOrigins"].MapTo(&trackOutputOrigins) = new PropTreeLib::Variables::PTLBoolean(false, "Writes file and line number to screen along with output");
            
            input["Memory"]["allowStackAllocation"].MapTo(&allowStackAllocation) = new PropTreeLib::Variables::PTLBoolean(false, "If set to true, will enable the emulated stack (use \"false\" for better debugging)");
        }
    };
}
#endif
