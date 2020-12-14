#ifndef CMF_BLOCK_VTK_H
#define CMF_BLOCK_VTK_H
#include "RefinementTreeNode.h"
#include "VtkFile.h"
#include <map>
#include <string>
#include <vector>
#include "AmrFcnTypes.h"
#include "Config.h"
#include "CmfError.h"
namespace cmf
{
    // Yes, this is a very lazy, silly implementation
    
    /// @brief A class for a double scalar
    /// @author WVN
    class BlockVtkDouble
    {
        public:
            /// @brief Constructor
            /// @param name_in The name of the scalar
            /// @author WVN
            BlockVtkDouble(std::string name_in) {name = name_in;}
            
            /// @brief Stream operator
            /// @param d The value to stream
            /// @author WVN
            void operator << (double d) {data.push_back(d);}
            
            /// @brief Returns the size of the data
            /// @author WVN
            size_t Size(void) {return data.size();}
            
            /// @brief Returns data[i]
            /// @param i Index
            /// @author WVN
            double operator [] (size_t i) {return data[i];}
            
        private:
            /// @brief The name of the scalar
            std::string name;
            
            /// @brief The data for this scalar
            std::vector<double> data;
    };
    
    /// @brief A quick-and-dirty class for outputting a vtk file containing blocks only, annotated by data. USED FOR DEBUGGING ONLY
    /// @author WVN
    class BlockVtk
    {
        public:
            /// @brief Constructor
            /// @author WVN
            BlockVtk(void){}
            
            /// @brief Destructor
            /// @author WVN
            ~BlockVtk(void)
            {
                for (std::map<std::string, BlockVtkDouble*>::iterator it = doubleScalars.begin(); it != doubleScalars.end(); it++)
                {
                    delete it->second;
                }
            }
            
            /// @brief Stream operator for adding blocks
            /// @param node The node to add
            /// @author WVN
            void operator << (RefinementTreeNode* node)
            {
                nodes.push_back(node);
            }
            
            /// @brief Indexing operator for retrieving scalars. Creates the scalar if it isn't found
            /// @param query The variable to fetch
            /// @author WVN
            BlockVtkDouble& operator [] (std::string query)
            {
                if (doubleScalars.find(query) == doubleScalars.end())
                {
                    doubleScalars.insert({query, new BlockVtkDouble(query)});
                }
                return *(doubleScalars[query]);
            }
            
            /// @brief Outputs data to VTK file
            /// @param filename The name of the file to be output
            /// @author WVN
            void Write(std::string filename)
            {
                VtkFile output(filename, VtkFormatType::ascii, VtkTopologyType::unstructuredGrid);
                int totalNumBlocks = nodes.size();
                for (std::map<std::string, BlockVtkDouble*>::iterator it = doubleScalars.begin(); it != doubleScalars.end(); it++)
                {
                    if (it->second->Size() != totalNumBlocks)
                    {
                        CmfError("A BlockVtk attempted to write to \"" + filename + "\", but the scalar field \"" + it->first + "\" is incomplete!");
                    }
                }
                output.Mesh()->Component("DATASET")->SetAttribute("numPoints",   (CMF_IS3D?8:4)*totalNumBlocks);
                output.Mesh()->Component("DATASET")->SetAttribute("bufferCount", 3*(CMF_IS3D?8:4)*totalNumBlocks);
                output.Mesh()->Component("DATASET")->SetAttribute("stride", 3);
                output.Mesh()->Component("CELLS")->SetAttribute("numPoints", totalNumBlocks);
                output.Mesh()->Component("CELLS")->SetAttribute("bufferCount", (CMF_IS3D?9:5)*totalNumBlocks);
                output.Mesh()->Component("CELLS")->SetAttribute("totalEntries", (CMF_IS3D?9:5)*totalNumBlocks);
                output.Mesh()->Component("CELLS")->SetAttribute("stride", (CMF_IS3D?9:5));
                output.Mesh()->Component("CELL_TYPES")->SetAttribute("numPoints", totalNumBlocks);
                output.Mesh()->Component("CELL_TYPES")->SetAttribute("bufferCount", totalNumBlocks);
                output.Mesh()->Component("CELL_TYPES")->SetAttribute("stride", 1);
                VtkBuffer points(output.Mesh()->Component("DATASET"));
                VtkBuffer edges(output.Mesh()->Component("CELLS"));
                VtkBuffer cellTypes(output.Mesh()->Component("CELL_TYPES"));
                int dummy = 0;
                for (int i = 0; i < totalNumBlocks; i++)
                {
                    nodes[i]->WriteBlockDataToVtkBuffers(points, edges, cellTypes, &dummy);
                }
                for (std::map<std::string, BlockVtkDouble*>::iterator it = doubleScalars.begin(); it != doubleScalars.end(); it++)
                {
                    VtkScalar* sBuf = output.CreateScalar(it->first, totalNumBlocks);
                    sBuf->Component("SCALARS")->SetAttribute("bufferCount", totalNumBlocks);
                    sBuf->Component("SCALARS")->SetAttribute("numPoints", totalNumBlocks);
                    sBuf->Component("SCALARS")->SetAttribute("stride", 1);
                    VtkBuffer scalarData(sBuf->Component("SCALARS"));
                    for (int i = 0; i < totalNumBlocks; i++)
                    {
                        scalarData << (*(it->second))[i];
                    }
                }
                output.Write();
            }
            
        private:

            /// @brief List of the integer scalars
            std::map<std::string, BlockVtkDouble*> doubleScalars;
            
            /// @brief The list of nodes
            std::vector<RefinementTreeNode*> nodes;
    };
}

#endif