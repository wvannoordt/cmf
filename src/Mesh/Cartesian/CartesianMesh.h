#ifndef CMF_CARTESIAN_MESH_H
#define CMF_CARTESIAN_MESH_H

#include <string>
#include <vector>
#include "Typedefs.h"
#include "AmrFcnTypes.h"
#include "PTL.h"
#include "RefinementTreeNode.h"
#include "BlockIterator.h"
#include "RefinementConstraint.h"
#include "RefinementBlock.h"
#include "ICmfMesh.h"
#include "CartesianMeshArrayHandler.h"
#include "ParallelGroup.h"
#include "CartesianMeshParallelPartition.h"
#include <initializer_list>
#include "BlockInfo.h"

namespace cmf
{    
    /// @brief A struct containing all the input information for a Cartesian mesh
    /// @author WVN
    struct CartesianMeshInputInfo : ICmfMeshInfo
    {
        /// @brief A vector of size CMF_DIM (at least) that represents the initial block dimensions
        std::vector<int> blockDim;

        /// @brief A vector of size 2*CMF_DIM (at least) that represents the block boundaries of the current refinement block as (xmin, xmax, ymin, ymax, [zmin, zmax])
        std::vector<double> blockBounds;
        
        /// @brief A logical vector of CMF_DIM indicating whether or not neighbor refinement constraints should be applied periodically in the corresponding direction [x, y, (z)]
        std::vector<bool> periodicRefinement;

        /// @brief See RefinementTreeNode::RefinementConstraint. Applied to all contained nodes
        RefinementConstraint::RefinementConstraint refinementConstraintType;

        /// @brief An array of size CMF_DIM that represents the initial block data dimensions
        std::vector<int> meshDataDim;
        
        /// @brief A struct containing information to create a parallel partition
        CartesianMeshParallelPartitionInfo partitionInfo;
        
        /// @brief A struct containing information to create a parallel partition
        CartesianMeshExchangeInfo exchangeInfo;

        /// @brief Constructor for the CartesianMeshInputInfo object.
        /// @param title_in title of the mesh
        /// @param inputTree PropertryTree to be read from
        /// @author WVN
        CartesianMeshInputInfo(PTL::PropertySection& inputTree) : ICmfMeshInfo(inputTree)
        {
            Define(*objectInput);
            Parse();
        }
        
        /// @brief Empy constructor for the manual definition
        /// @author WVN
        CartesianMeshInputInfo(void)
        {
            
        }

        /// @brief Defines the object from the input secton
        /// @param input The section to be read from
        /// @author WVN
        void Define(PTL::PropertySection& input)
        {
            input["blockDim"].MapTo(&blockDim) = new PTL::PTLIntegerVector("Base block dimensions", 3, [](int i){if (i==2 && CMF_DIM==2) {return 1;} return 2;});
            
            input["blockBounds"].MapTo(&blockBounds) = new PTL::PTLDoubleVector("Base block bounds", 6, [](int i){return (double)(i&1);});
            
            input["refinementConstraintType"].MapTo((int*)&refinementConstraintType)
                = new PTL::PTLAutoEnum(RefinementConstraint::factor2CompletelyConstrained, RefinementConstraintStr, "Determines how refinements are constrained");
                
            input["periodicRefinement"].MapTo(&periodicRefinement)
                = new PTL::PTLBooleanVector("Determines if the refinementConstraintType applies accross domain boundaries", 3, [](int i){return true;});
                
            input["meshDataDim"].MapTo(&meshDataDim) = new PTL::PTLIntegerVector("Dimensions of data", 3, [](int i){return 16;});
            
            partitionInfo.Define(input["Partition"]);
            exchangeInfo.Define(input["Exchanges"]);
        }
    };

    /// @brief Mesh class for cartesian grids
    /// @author WVN
    class CartesianMesh : public ICmfMesh, public IBlockIterable
    {
        friend class CartesianMeshArrayHandler;
        friend class CartesianMeshArray;
        public:
            /// @brief Constructor for the CartesianMesh object.
            /// @param input Input data for the
            /// @author WVN
            CartesianMesh(CartesianMeshInputInfo input);

            /// @brief Destructor for the CartesianMesh object.
            /// @author WVN
            ~CartesianMesh(void);

            /// @brief Returns the underlying RefinementBlock object
            /// @author WVN
            RefinementBlock* Blocks(void);

            /// @brief Returns the mesh array handler
            /// @author WVN
            ICmfMeshArrayHandler* GetArrayHandler(void);

            /// @brief Defines a variable with the given name
            /// @param name The name of the variable
            /// @author WVN
            CartesianMeshArray& DefineVariable(std::string name);
            
            /// @brief Defines a variable with the given name
            /// @param name The name of the variable
            /// @param filter A block filter determining whether the block lies in the domain of the variable
            /// @author WVN
            CartesianMeshArray& DefineVariable(std::string name, NodeFilter_t filter);
            
            /// @brief Defines a variable with the given name
            /// @param name The name of the variable
            /// @param elementType the type of a single element
            /// @param filter A block filter determining whether the block lies in the domain of the variable
            /// @author WVN
            CartesianMeshArray& DefineVariable(std::string name, CmfArrayType::CmfArrayType elementType, NodeFilter_t filter);
            
            /// @brief Defines a variable with the given name
            /// @param name The name of the variable
            /// @param elementType the type of a single element
            /// @param arrayDimensions the dimensions of the array (per element)
            /// @param filter A block filter determining whether the block lies in the domain of the variable
            /// @author WVN
            CartesianMeshArray& DefineVariable(std::string name, CmfArrayType::CmfArrayType elementType, std::initializer_list<int> arrayDimensions, NodeFilter_t filter);
            
            /// @brief Defines a variable with the given name
            /// @param name The name of the variable
            /// @param elementType the type of a single element
            /// @param arrayDimensions the dimensions of the array (per element)
            /// @param filter A block filter determining whether the block lies in the domain of the variable
            /// @author WVN
            CartesianMeshArray& DefineVariable(std::string name, CmfArrayType::CmfArrayType elementType, std::vector<int> arrayDimensions, NodeFilter_t filter);
            
            /// @brief Defines a variable with the given name
            /// @param name The name of the variable
            /// @param elementType the type of a single element
            /// @author WVN
            CartesianMeshArray& DefineVariable(std::string name, CmfArrayType::CmfArrayType elementType);
            
            /// @brief Defines a variable with the given name
            /// @param name The name of the variable
            /// @param elementType the type of a single element
            /// @param arrayDimensions the dimensions of the array (per element)
            /// @param filter A block filter determining whether the block lies in the domain of the variable
            /// @author WVN
            CartesianMeshArray& DefineVariable(std::string name, CmfArrayType::CmfArrayType elementType, std::initializer_list<int> arrayDimensions);
            
            /// @brief Defines a variable with the given name
            /// @param info An ArrayInfo struct defining the properties of the array
            /// @author WVN
            CartesianMeshArray& DefineVariable(ArrayInfo info);
            
            /// @brief Defines a variable with the given name
            /// @param info An ArrayInfo struct defining the properties of the array
            /// @param filter A block filter determining whether the block lies in the domain of the variable
            /// @author WVN
            CartesianMeshArray& DefineVariable(ArrayInfo info, NodeFilter_t filter);
            
            /// @brief Defines a variable and fills it with the specified Cartesian coordinate
            /// @param filter A block filter determining whether the block lies in the domain of the variable
            /// @param direction The corrdinate to generate (0=x, 1=y, 2=z)
            /// @author WVN
            CartesianMeshArray& CreateCoordinateVariable(NodeFilter_t filter, int direction);
            
            /// @brief Defines a variable and fills it with the specified Cartesian coordinate
            /// @param direction The corrdinate to generate (0=x, 1=y, 2=z)
            /// @author WVN
            CartesianMeshArray& CreateCoordinateVariable(int direction);
            
            /// @brief Returns a BlockInfo struct computed from the given block
            /// @param node The block to have info returned for
            /// @author WVN
            BlockInfo GetBlockInfo(RefinementTreeNode* node);
            
            /// @brief Returns a BlockInfo struct computed from the given block
            /// @param blockIter A block iterator that provides a block to get information for
            /// @author WVN
            BlockInfo GetBlockInfo(BlockIterator& blockIter);
            
            /// @brief Returns the bounding box of the mesh
            /// @author WVN
            std::vector<double> GetBlockBounds(void) {return blockBounds;}
            
            /// @brief Returns the total number of nodes that are contained within the iterable object
            /// @author WVN
            size_t Size(void);
            
            /// @author WVN
            /// @brief Gets the list of blocks to be iterated over
            virtual std::vector<RefinementTreeNode*>* GetAllNodes(void) override;
            
            /// @brief Begin() overload for range iteration
            /// @author WVN
            std::vector<RefinementTreeNode*>::iterator begin() noexcept;
            
            /// @brief End() overload for range iteration
            /// @author WVN
            std::vector<RefinementTreeNode*>::iterator end() noexcept;
            
            /// @author WVN
            /// @brief Returns true if the mesh has this node in its parallel partition
            /// @param node The node to check
            bool ParallelPartitionContainsNode(RefinementTreeNode* node);
            
            /// @brief \see IBlockIterable::GetRefinementBlockObject
            /// @author WVN
            RefinementBlock* GetRefinementBlockObject(void);
            
            /// @brief Throws an error if the block structure varies between ranks on the relevant
            /// ParallelGroup \see ParallelGroup
            /// @author WVN
            void AssertSynchronizeBlocks(void);
            
            /// @author WVN
            /// @brief Returns the title of the mesh
            std::string GetTitle(void);
            
            /// @author WVN
            /// @brief Returns a mesh array with the given name
            /// @param name The name of the array to fetch
            CartesianMeshArray& operator [] (std::string name);
            
            /// @author WVN
            /// @brief Returns the parallel partition object
            CartesianMeshParallelPartition* GetPartition(void);
            
            /// @brief Returns the minimum grid spacing over the whole mesh
            /// @author WVN
            double GetMinimumSpacing(void);
            
            /// @brief Reads the data from the parallel file
            /// @param file The file to read from
            /// @author WVN
            virtual void ReadFromFile(ParallelFile& file) override final;
            
            /// @brief Writes the data to the parallel file
            /// @param file The file to write to
            /// @author WVN
            virtual void WriteToFile(ParallelFile& file) override final;
            
            /// @brief Returns the dimensions of each block in cells (without exchanges)
            /// @author WVN
            std::vector<int> GetMeshDataDim(void) { return meshDataDim; }
            
            /// @brief Returns the dimensions of exchange cells of each block
            /// @author WVN
            std::vector<int> GetExchangeDim(void) { return exchangeDim; }

        private:
            
            /// @author WVN
            /// @brief Creates a parallel partition for the current mesh
            /// @param partitionInfo the input struct containing parameters to create the paralle partition
            CartesianMeshParallelPartition* CreateParallelPartition(CartesianMeshParallelPartitionInfo& partitionInfo);

            /// @brief contains the mesh blocks of the current mesh
            RefinementBlock* blocks;

            /// @brief An vector that represents the initial block dimensions
            std::vector<int> blockDim;

            /// @brief An array of size 2*CMF_DIM the represents the block boundaries of the current refinement block as (xmin, xmax, ymin, ymax, [zmin, zmax])
            std::vector<double> blockBounds;

            /// @brief An array of size CMF_DIM that represents the initial block data dimensions
            std::vector<int> meshDataDim;
            
            /// @brief An array of size CMF_DIM that represents the number of exchange cells (used to facilitate data exchange) in each direction
            std::vector<int> exchangeDim;

            /// @brief See RefinementTreeNode::RefinementConstraint. Applied to all contained nodes
            RefinementConstraint::RefinementConstraint refinementConstraintType;
            
            /// @brief A handler object used to define arrays over the current mesh
            CartesianMeshArrayHandler* arrayHandler;
            
            /// @brief Indicates whether the mesh has been partitioned
            bool hasParallelPartition;
            
            /// @brief Parallel partition object for the current mesh
            CartesianMeshParallelPartition* partition;
        
            friend class CartesianMeshParallelPartition;
            friend class CartesianMeshExchangeHandler;
            template <typename arType, const int elementRank> friend struct BlockArray;
            friend struct RawUnwrappedBlockArray;
    };
}

#endif
