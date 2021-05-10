#include "CartesianMeshArrayParallelVtkWriter.h"
#include "CartesianMeshArrayHandler.h"
#include "CartesianMesh.h"
#include "BlockArray.h"
#include "Path.h"
#include "Utils.hx"
#include "Base64ByteConversionStream.h"

namespace cmf
{
    CartesianMeshArrayParallelVtkWriter::CartesianMeshArrayParallelVtkWriter(std::string directory_in, std::string fileTitle_in)
    {
        outputDirectory = directory_in;
        fileTitle = fileTitle_in;
    }
    
    CartesianMeshArrayParallelVtkWriter::~CartesianMeshArrayParallelVtkWriter(void)
    {
        
    }
    
    std::string CartesianMeshArrayParallelVtkWriter::GetMetaFileName(CartesianMeshArray& array)
    {
        //Create path for the meta-file
        Path outputPathMetaFile(outputDirectory);
        std::string str1 = fileTitle + ".vtm";
        outputPathMetaFile += str1;
        std::string filenameMetaFile = outputPathMetaFile.Str();
        return filenameMetaFile;
    }
    
    std::string CartesianMeshArrayParallelVtkWriter::GetBlockFileTemplate(CartesianMeshArray& array)
    {
        //Create prototype path for the  block outputs
        Path outputPathBlockPrototype(outputDirectory);
        std::string str1 = fileTitle + "_bk{}.vtr";
        outputPathBlockPrototype += str1;
        std::string blockTemplateFileName = outputPathBlockPrototype.Str();
        return blockTemplateFileName;
    }
    
    std::string CartesianMeshArrayParallelVtkWriter::GetBlockFileTemplateRelativeToMetaFile(CartesianMeshArray& array)
    {
        //Create prototype path for the  block outputs
        std::string str1 = fileTitle + "_bk{}.vtr";
        return str1;
    }
    
    Vec3<int> CartesianMeshArrayParallelVtkWriter::GetLargestBlockDims(CartesianMeshArray& array)
    {
        Vec3<int> output(0, 0, 0);
        for (auto lb: array)
        {
            auto info = array.GetBlockInfo(lb);
            int nCellsi = info.imax - info.imin;
            int nCellsk = info.kmax - info.kmin;
            int nCellsj = info.jmax - info.jmin;
            int nGuardi = info.exchangeI;
            int nGuardj = info.exchangeJ;
            int nGuardk = info.exchangeK;
            int nTotali = nCellsi + 2*nGuardi;
            int nTotalj = nCellsj + 2*nGuardj;
            int nTotalk = nCellsk + 2*nGuardk;
            output[0] = CMFMAX(output[0], nTotali);
            output[1] = CMFMAX(output[1], nTotalj);
            output[2] = CMFMAX(output[2], nTotalk);
        }
        return output;
    }
    
    //This is a bit of a mess at the moment
    void CartesianMeshArrayParallelVtkWriter::Export(CartesianMeshArray& array)
    {
        //create comma-separated list of the variable components
        auto vnames = array.GetComponentNames();
        std::string varsString = "";
        for (int l = 0; l < vnames.size(); l++)
        {
            varsString += ((l==0)?"":",");
            varsString += vnames[l];
        }
        
        std::string filenameMetaFile              = this->GetMetaFileName(array);
        std::string blockTemplateFileName         = this->GetBlockFileTemplate(array);
        std::string blockTemplateFileNameRelative = this->GetBlockFileTemplateRelativeToMetaFile(array);
        
        //Get the parallel group
        auto parGroup = array.Mesh()->GetGroup();
        int blocksWrittenByMe = 0;
        
        //We will allocate a temporary array to store the block data in,
        //since the byte storage order is different than for the output
        //Allocate to accommodate the size of the largest block
        Vec3<int> maxSize = this->GetLargestBlockDims(array);
        size_t elementSize = SizeOfArrayType(array.GetElementType());
        MdArray<char, 4> bufferBlock(elementSize, maxSize[0], maxSize[1], maxSize[2]);
        bufferBlock.data = (char*)Cmf_Alloc(elementSize*maxSize[0]*maxSize[1]*maxSize[2]*sizeof(char));
        
        
        int numBlocksWritten = 0;
        int numBlocksLocal = 0;
        for (auto lb: array)
        {
            numBlocksLocal++;
        }
        int* allBlocks = parGroup->SharedValues(numBlocksLocal);
        int blocksBefore = 0;
        int totalNumBlocksWritten = parGroup->Sum(numBlocksLocal);
        for (int p = 0; p < parGroup->Rank(); p++)
        {
            blocksBefore += allBlocks[p];
        }
        
        for (int process = 0; process < parGroup->Size(); process++)
        {
            if (parGroup->Rank()==process)
            {
                for (auto lb: array)
                {
                    
                    auto info = array.GetBlockInfo(lb);
                    
                    int nCellsi = info.imax - info.imin;
                    int nCellsk = info.kmax - info.kmin;
                    int nCellsj = info.jmax - info.jmin;
                    int nGuardi = info.exchangeI;
                    int nGuardj = info.exchangeJ;
                    int nGuardk = info.exchangeK;
                    int nTotali = nCellsi + 2*nGuardi;
                    int nTotalj = nCellsj + 2*nGuardj;
                    int nTotalk = nCellsk + 2*nGuardk;
                    
                    std::string filename = strformat(blockTemplateFileName, ZFill(blocksWrittenByMe+blocksBefore, 7));
                    double bds[6] = {0.0};
                    for (int i = 0; i < 2*CMF_DIM; i++) bds[i] = info.blockBounds[i];
                    double ghostBnds[6] = {0.0};
                    for (int i = 0; i < CMF_DIM; i++)
                    {
                        ghostBnds[2*i] = info.blockBounds[2*i] - info.exchangeDim[i]*info.dx[i];
                        ghostBnds[2*i+1] = info.blockBounds[2*i+1] + info.exchangeDim[i]*info.dx[i];
                    }
                    
                    std::ofstream myfile;
                    myfile.open(filename.c_str());
                    std::string sp20 = spaces(20);
                    const char* csp20 = sp20.c_str();
                    std::string endianness = MachineIsBigEndian()?"BigEndian":"LittleEndian";
                    myfile << "<?xml version=\"1.0\"?>\n";
                    myfile << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"" << endianness << "\" header_type=\"UInt32\">" << std::endl;
                    myfile << spaces(4) << strformat("<RectilinearGrid WholeExtent=\"0 {} 0 {} 0 {}\">", nTotali, nTotalj, nTotalk) << std::endl;
                    myfile << spaces(8) << "<FieldData>" << std::endl;
                    myfile << spaces(12) << "<DataArray type=\"Int32\" Name=\"avtRealDims\" NumberOfTuples=\"6\" format=\"ascii\">" << std::endl;
                    myfile << spaces(16) << strformat("{} {} {} {} {} {}", nGuardi, nGuardj+nCellsi, nGuardj, nGuardj+nCellsj, nGuardk, CMF_IS3D?(nGuardk+nCellsk):0) << std::endl;
                    myfile << spaces(12) << "</DataArray>" << std::endl;
                    myfile << spaces(12) << "<DataArray type=\"Float64\" Name=\"avtOriginalBounds\" NumberOfTuples=\"6\" format=\"ascii\">" << std::endl;
                    myfile << spaces(16) << strformat("{} {} {} {} {} {}", bds[0], bds[1], bds[2], bds[3], bds[4], bds[5]) << std::endl;
                    myfile << spaces(12) << "</DataArray>" << std::endl;
                    myfile << spaces(8) << "</FieldData>" << std::endl;
                    myfile << spaces(8) << strformat("<Piece Extent=\"0 {} 0 {} 0 {}\">", nTotali, nTotalj, nTotalk) << std::endl;
                    
                    RawUnwrappedBlockArray blockBytes(elementSize, vnames.size(), array[lb]);
                    
                    myfile << spaces(12) << "<CellData Scalars=\"" << varsString << "\">" << std::endl;
                    size_t storageIdx = 0;
                    for (int var = 0; var < vnames.size(); var++)
                    {
                        myfile << spaces(16) << "<DataArray type=\"Float64\" Name=\"" << vnames[var] << "\" format=\"binary\">" << std::endl;
                        for (cmf::cell_t k = info.kmin-info.exchangeK; k < info.kmax+info.exchangeK; k++)
                        {
                            for (cmf::cell_t j = info.jmin-info.exchangeJ; j < info.jmax+info.exchangeJ; j++)
                            {
                                for (cmf::cell_t i = info.imin-info.exchangeI; i < info.imax+info.exchangeI; i++)
                                {
                                    for (int y = 0; y < elementSize; y++)
                                    {
                                        bufferBlock(y, i+info.exchangeI, j+info.exchangeJ, k+info.exchangeK) = blockBytes(y, var, i, j, k);
                                    }
                                }
                            }
                        }
                        ////https://mathema.tician.de/what-they-dont-tell-you-about-vtk-xml-binary-formats/
                        unsigned int ss = bufferBlock.totalSize;
                        Base64ByteConversionStream(myfile, (char*)(&ss), sizeof(int));
                        Base64ByteConversionStream(myfile, bufferBlock.data, bufferBlock.totalSize);
                        myfile << "\n" << spaces(16) << "</DataArray>" << std::endl;
                    }
                    myfile << spaces(16) << "<DataArray type=\"UInt8\" Name=\"avtGhostZones\" format=\"ascii\">" << std::endl;
                    auto isGhost = [&](int i, int j, int k) -> bool {return (i<0)||(i>=nCellsi)||(j<0)||(j>=nCellsj)||(k<0)||(k>=nCellsk);};
                    for (cmf::cell_t k = info.kmin-info.exchangeK; k < info.kmax+info.exchangeK; k++)
                    {
                        for (cmf::cell_t j = info.jmin-info.exchangeJ; j < info.jmax+info.exchangeJ; j++)
                        {
                            for (cmf::cell_t i = info.imin-info.exchangeI; i < info.imax+info.exchangeI; i++)
                            {
                                myfile << csp20 << (isGhost(i, j, k)?16:0) << "\n";
                            }
                        }
                    }
                    myfile << spaces(16) << "</DataArray>" << std::endl;
                    myfile << spaces(12) << "</CellData>" << std::endl;
                    myfile << spaces(12) << "<Coordinates>" << std::endl;
                    myfile << spaces(16) << strformat("<DataArray type=\"Float64\" format=\"ascii\" RangeMin=\"{}\" RangeMax=\"{}\">", ghostBnds[0], ghostBnds[1]) << std::endl;
                    for (int i = -nGuardi; i <=nCellsi+nGuardi; i++)
                    {
                        myfile << csp20 << info.blockBounds[0] + i*info.dx[0] << "\n";
                    }
                    myfile << spaces(16) << "</DataArray>" << std::endl;
                    myfile << spaces(16) << strformat("<DataArray type=\"Float64\" format=\"ascii\" RangeMin=\"{}\" RangeMax=\"{}\">", ghostBnds[2], ghostBnds[3]) << std::endl;
                    for (int j = -nGuardj; j <=nCellsj+nGuardj; j++)
                    {
                        myfile << csp20 << info.blockBounds[2] + j*info.dx[1] << "\n";
                    }
                    myfile << spaces(16) << "</DataArray>" << std::endl;
                    myfile << spaces(16) << strformat("<DataArray type=\"Float64\" format=\"ascii\" RangeMin=\"{}\" RangeMax=\"{}\">", ghostBnds[4], ghostBnds[4]) << std::endl;
                    for (int k = -nGuardk; k <=nCellsk+nGuardk; k++)
                    {
#if(CMF_IS3D)
                        myfile << csp20 << info.blockBounds[4] + k*info.dx[2] << "\n";
#else
                        myfile << csp20 << 0.0 << "\n";
#endif
                    }
                    myfile << spaces(16) << "</DataArray>" << std::endl;
                    myfile << spaces(12) << "</Coordinates>" << std::endl;
                    myfile << spaces(8) << "</Piece>" << std::endl;
                    myfile << spaces(4) << "</RectilinearGrid>" << std::endl;
                    myfile << "</VTKFile>" << std::endl;
                    
                    
                    myfile.close();
                    blocksWrittenByMe++;
                }
            }
        }
        
        if (parGroup->IsRoot())
        {
            std::string filename = filenameMetaFile;
            std::ofstream myfile;
            myfile.open(filename.c_str());
            myfile << "<?xml version=\"1.0\"?>\n";
            myfile << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\">" << std::endl;
            myfile << spaces(4) << "<vtkMultiBlockDataSet>" << std::endl;
            myfile << spaces(8) << "<Block index =\"0\">" << std::endl;
            for (int b = 0; b < totalNumBlocksWritten; b++)
            {
                std::string blockFileName = strformat(blockTemplateFileNameRelative, ZFill(b, 7));
                myfile << spaces(12) << strformat("<DataSet index=\"{}\" file=\"{}\"/>", b, blockFileName) << std::endl;
            }
            myfile << spaces(8) << "</Block>" << std::endl;
            myfile << spaces(4) << "</vtkMultiBlockDataSet>" << std::endl;
            myfile << "</VTKFile>" << std::endl;
            myfile.close();
        }
        Cmf_Free(bufferBlock.data);
        parGroup->Synchronize();
    }
}