#ifndef CMF_CART_MESH_ARRAY_EXCHANGE_HANDLER_H
#define CMF_CART_MESH_ARRAY_EXCHANGE_HANDLER_H
#include "CmfError.h"
#include "CmfScreen.h"
#include "ICmfInputObject.h"
namespace cmf
{
    class CartesianMesh;
    struct CartesianMeshExchangeInfo : public ICmfInputObject
    {
        /// @brief The order of interpolation for exchanges of differing levels
        int interpolationOrder;
        
        /// @brief An array of size CMF_DIM that represents the number of exchange cells (used to facilitate data exchange) in each direction
        int* exchangeDim;
        
        /// @brief empty Constructor
        /// @author WVN
        CartesianMeshExchangeInfo(void){exchangeDim = NULL;}
        
        /// @brief Constructor for the CartesianMeshInputInfo object.
        /// @param inputSection Section to be read from
        /// @author WVN
        CartesianMeshExchangeInfo(PTL::PropertySection& inputSection) : ICmfInputObject(inputSection)
        {
            Define(*objectInput);
            Parse();
        }
        
        /// @brief Defines the object from the input secton
        /// @param input The section to be read from
        /// @author WVN
        void Define(PTL::PropertySection& input)
        {
            input["interpolationOrder"].MapTo(&interpolationOrder)
                = new PTL::PTLInteger(2, "The order of interpolation from one level to the next, or previous, when performing exchanges");
            input["exchangeDim"].MapTo(&exchangeDim) = new PTL::PTLStaticIntegerArray(CMF_DIM, "Number of exchange cells to use in each direction", [](int i){return 1;});
        }
    };
    /// @brief Defines an object for handling data exchanges on Cartesian meshes
    /// @author WVN
    class CartesianMeshExchangeHandler
    {
        public:
            /// @brief Constructor
            /// @param mesh_in The mesh over which the exchanges are defined
            /// @param inputInfo The info object that contains constructor parameters
            /// @author WVN
            CartesianMeshExchangeHandler(CartesianMesh* mesh_in, CartesianMeshExchangeInfo& inputInfo);
            
            /// @brief Destructor
            /// @author WVN
            ~CartesianMeshExchangeHandler(void);
            
        private:
            /// @param mesh The mesh over which the exchanges are defined
            CartesianMesh* mesh;
            
            /// @brief The order of interpolation for exchanges of differing levels
            int interpolationOrder;
            
            /// @brief An array of size CMF_DIM that represents the number of exchange cells (used to facilitate data exchange) in each direction
            int* exchangeDim;
    };
}

#endif