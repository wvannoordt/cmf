#include "SerialCartesianVtk.h"
#include "CmfScreen.h"
#include "Utils.hx"
#include "StringUtils.h"
#include "BlockIndexing.h"
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
    
    void SerialCartesianVtk::Write(void)
    {
        output->Write();
    }
    
    SerialCartesianVtk& SerialCartesianVtk::operator << (CartesianMeshArray& variable)
    {
        WriteLine(5, "Stream variable \"" + variable.GetVarName() + "\"");
        VtkScalar* newScalar = output->CreateScalar(variable.GetVarName(), totalNumPointsOnMesh);
        newScalar->Component("SCALARS")->SetAttribute("bufferCount", totalNumPointsOnMesh);
        newScalar->Component("SCALARS")->SetAttribute("numPoints", totalNumPointsOnMesh);
        newScalar->Component("SCALARS")->SetAttribute("stride", 1);
        VtkBuffer scalarData(newScalar->Component("SCALARS"));
        int idx[CMF_DIM];
        for (BlockIterator lb(mesh, filter); lb.HasNext(); lb++)
        {
            BlockInfo info = mesh->GetBlockInfo(lb);
            double* buf = (double*)variable[lb];
            cmf_pkloop(idx[2], 0, info){cmf_pjloop(idx[1], 0, info){cmf_piloop(idx[0], 0, info){
                scalarData << buf[cmf_idx(idx[0], idx[1], idx[2], info)];
            }}}
        }
        return *this;
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
        totalNumPointsOnMesh = totalNumPoints;
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
        VtkBuffer points(output->Mesh()->Component("DATASET"));
        VtkBuffer edges(output->Mesh()->Component("CELLS"));
        VtkBuffer cellTypes(output->Mesh()->Component("CELL_TYPES"));
        double x[CMF_DIM];
        int idx[CMF_DIM];
        int ppp = 0;
        double cellBounds[2*CMF_DIM];
        double cellCorner[CMF_DIM];
        int currentPointIdx = 0;
        for (BlockIterator lb(mesh, filter); lb.HasNext(); lb++)
        {
            BlockInfo info = mesh->GetBlockInfo(lb);
            int localPoints = 1;
            __dloop(localPoints*=info.totalDataDim[d]);
            cmf_pkloop(idx[2], 0, info){cmf_pjloop(idx[1], 0, info){cmf_piloop(idx[0], 0, info){
                cellTypes << ((CMF_IS3D)?(VtkCellType::voxel):(VtkCellType::pixel));
                edges << (CMF_IS3D?8:4);
                __dloop(x[d] = info.blockBounds[2*d]+((double)idx[d] + 0.5)*info.dx[d]);
                __dloop(cellBounds[2*d]   = x[d] - 0.5*info.dx[d]);
                __dloop(cellBounds[2*d+1] = x[d] + 0.5*info.dx[d]);
                for (int cornerIdx = 0; cornerIdx<(1<<CMF_DIM); cornerIdx++)
                {
                    __dloop(cellCorner[d] = cellBounds[2*d+((cornerIdx>>d)&1)]);
                    __dloop(points << cellCorner[d]);
#if(!CMF_IS3D)
                    points << 0.0;
#endif
                    edges << currentPointIdx;
                    currentPointIdx++;
                }
            }}}
        }
    }
}