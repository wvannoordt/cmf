#include "SerialCartesianVtk.h"
#include "CmfScreen.h"
#include "Utils.hx"
#include "StringUtils.h"
namespace cmf
{
    SerialCartesianVtk::SerialCartesianVtk(CartesianMesh& mesh_in, std::string filename_in, NodeFilter_t filter_in)
    {
        Build(mesh_in, filename_in, filter_in);
    }
    
    SerialCartesianVtk::SerialCartesianVtk(CartesianMesh& mesh_in, std::string filename_in)
    {
        Build(mesh_in, filename_in, BlockFilters::Terminal);
    }
    
    SerialCartesianVtk::~SerialCartesianVtk(void)
    {
        Destroy();
    }
    
    void SerialCartesianVtk::Destroy(void)
    {
        if (requireDelete)
        {
            delete output;
        }
    }
    
    void SerialCartesianVtk::Build(CartesianMesh& mesh_in, std::string filename_in, NodeFilter_t filter_in)
    {
        requireDelete = false;
        mesh = &mesh_in;
        filename = filename_in;
        filter = filter_in;
        output = new VtkFile(filename, VtkFormatType::ascii, VtkTopologyType::unstructuredGrid);
        requireDelete = true;
        ProcessBlocks();
    }
    
    void SerialCartesianVtk::ProcessBlocks(void)
    {
        size_t totalNumPoints = 0;
        int numBlocks = 0;
        int i = 0;
        for (BlockIterator lb(mesh, filter); lb.HasNext(); lb++)
        {
            blocks.push_back(lb.Node());
            BlockInfo info = mesh->GetBlockInfo(lb);
            int localPoints = 1;
            __dloop(localPoints*=info.totalDataDim[d]);
            numBlocks++;
            totalNumPoints += localPoints;
        }
        WriteLine(5, "SerialCartesianVtk found " + NiceCommaString(totalNumPoints) + " points over " + NiceCommaString(numBlocks) + " blocks");
        output->Mesh()->Component("DATASET")->SetAttribute("numPoints",   (CMF_IS3D?8:4)*totalNumPoints);
        output->Mesh()->Component("DATASET")->SetAttribute("bufferCount", 3*(CMF_IS3D?8:4)*totalNumPoints);
        output->Mesh()->Component("DATASET")->SetAttribute("stride", 3);
        output->Mesh()->Component("CELLS")->SetAttribute("numPoints", totalNumPoints);
        output->Mesh()->Component("CELLS")->SetAttribute("bufferCount", (CMF_IS3D?9:5)*totalNumPoints);
        output->Mesh()->Component("CELLS")->SetAttribute("totalEntries", (CMF_IS3D?9:5)*totalNumPoints);
        output->Mesh()->Component("CELLS")->SetAttribute("stride", (CMF_IS3D?9:5));
        output->Mesh()->Component("CELL_TYPES")->SetAttribute("numPoints", totalNumPoints);
        output->Mesh()->Component("CELL_TYPES")->SetAttribute("bufferCount", totalNumPoints);
        output->Mesh()->Component("CELL_TYPES")->SetAttribute("stride", 1);
        ///START HERE
    }
}