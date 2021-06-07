#ifndef CMF_CART_MESH_ARRAY_EXCHANGE_HANDLER_H
#define CMF_CART_MESH_ARRAY_EXCHANGE_HANDLER_H
#include "CmfError.h"
#include "CmfScreen.h"
#include "ICmfInputObject.h"
#include "RefinementTreeNode.h"
#include "DataExchangePattern.h"
#include "CartesianMeshArray.h"
#include "DebugPointCloud.h"
#include <map>

namespace cmf
{
    struct ExchangeContextBlockData;
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
            input["exchangeDim"].MapTo(&exchangeDim) = new PTL::PTLStaticIntegerArray(CMF_DIM, "Number of exchange cells to use in each direction", [](int i){return 0;});
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
            
            /// @brief Creates a new parallel exchange pattern for the provided array
            /// @param meshArray The array to create an exchange pattern for
            /// @author WVN
            DataExchangePattern* CreateMeshArrayExchangePattern(CartesianMeshArray* meshArray);
            
        private:
            /// @brief Creates a new parallel exchange pattern for the provided array
            /// @param meshArray The array to define an exchange pattern for
            /// @param pattern The exchange pattern to be defined
            /// @author WVN
            void DefineExchangePatternsForArray(CartesianMeshArray* meshArray, DataExchangePattern* pattern);
            
            /// @brief Creates an exchange transaction between two neighboring nodes
            /// @param pattern The exchange pattern to add the transaction to
            /// @param meshArray The array to define an exchange pattern for
            /// @param currentNode The current node sending information
            /// @param neighborNode The node that will receive information
            /// @param relationship The senderNode -> receiverNode relationship
            /// @author WVN
            void CreateExchangeTransaction
                (
                    DataExchangePattern* pattern,
                    CartesianMeshArray* meshArray,
                    RefinementTreeNode* currentNode,
                    RefinementTreeNode* neighborNode, 
                    NodeEdge relationship
                );
            
            /// @brief Creates a direct-injection transaction between two neighboring nodes
            /// @param pattern The exchange pattern to add the transaction to
            /// @param meshArray The array to define an exchange pattern for
            /// @param currentInfo Contains contextual information about the current node
            /// @param neighborInfo Contains contextual information about the neighbor node
            /// @param relationship The senderNode -> receiverNode relationship
            /// @author WVN
            void CreateDirectInjectionTransaction
                (
                    DataExchangePattern* pattern,
                    CartesianMeshArray* meshArray,
                    ExchangeContextBlockData& currentInfo,
                    ExchangeContextBlockData& neighborInfo,
                    NodeEdge& relationship
                );
                
            /// @brief Creates a general interpolating transaction between two neighboring nodes
            /// @param pattern The exchange pattern to add the transaction to
            /// @param meshArray The array to define an exchange pattern for
            /// @param currentInfo Contains contextual information about the current node
            /// @param neighborInfo Contains contextual information about the neighbor node
            /// @param relationship The senderNode -> receiverNode relationship
            /// @author WVN
            void CreateGeneralExchangePattern
                (
                    DataExchangePattern* pattern,
                    CartesianMeshArray* meshArray,
                    ExchangeContextBlockData& currentInfo,
                    ExchangeContextBlockData& neighborInfo,
                    NodeEdge& relationship
                );
            
            /// @brief Returns vectors describing an exchange region in the current block's index coordinate system
            /// @param currentInfo Contains contextual information about the current node
            /// @param neighborInfo Contains contextual information about the neighbor node
            /// @param edgeVector The edge vector from the current node to the neighbor node
            /// @param exchangeRegionOut (output) the bounding box (in index space) of the exchange region
            /// @param exchangeSizeOut (output) The number of cells in the exchange region
            /// @author WVN
            void GetExchangeRegionInLocalIndexCoordinates(ExchangeContextBlockData& currentInfo, ExchangeContextBlockData& neighborInfo, Vec3<int> edgeVector, Vec<double, 6>& exchangeRegionOut, Vec3<int>& exchangeSizeOut);
            
            /// @brief Takes the provided region in the current node's index space and maps it into the neighbor node's index space
            /// @param currentInfo Contains contextual information about the current node
            /// @param neighborInfo Contains contextual information about the neighbor node
            /// @param edgeVector The edge vector from the current node to the neighbor node
            /// @param exchangeRegionCurrentView the bounding box (in index space) of the exchange region as seen by the current node
            /// @param exchangeRegionNeighborView (output) the bounding box (in index space) of the exchange region as seen by the neighbor node
            /// @author WVN
            void MapExchangeRegionIntoNeighborIndexCoordinates
                ( 
                    ExchangeContextBlockData& currentInfo,
                    ExchangeContextBlockData& neighborInfo,
                    Vec3<int> edgeVector,
                    Vec<double, 6>& exchangeRegionCurrentView,
                    Vec<double, 6>& exchangeRegionNeighborView
                );
            
            /// @brief mesh The mesh over which the exchanges are defined
            CartesianMesh* mesh;
            
            /// @brief The order of interpolation for exchanges of differing levels
            int interpolationOrder;
            
            /// @brief An array of size CMF_DIM that represents the number of exchange cells (used to facilitate data exchange) in each direction
            int* exchangeDim;
        
            /// @brief Map of block exchange patterns
            std::map<CartesianMeshArray*, DataExchangePattern*> exchanges;
    };
}

#endif