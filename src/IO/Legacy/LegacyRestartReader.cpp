#include "LegacyRestartReader.h"
#include "CmfPrint.h"
#include "CmfScreen.h"
#include "StringUtils.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include "BinaryFile.h"
#include <math.h>
#include <map>

namespace cmf
{
    LegacyRestartReader::LegacyRestartReader(std::string interpolationInfo_in,  std::string blockInfo_in, std::string flowData_in)
    {
        interpolationInfoFilename = interpolationInfo_in;
        blockInfoFilename = blockInfo_in;
        flowDataFilename = flowData_in;
    }
    
    CartesianMeshArray& LegacyRestartReader::LoadToMesh(CartesianMesh& domain)
    {
        LegacyRestartBlockArrangement blocks(blockInfoFilename);
        for (int level = 1; level <= blocks.maxLevel; level++)
        {
            std::map<RefinementTreeNode*, char> refines;
            for (size_t lb = 0; lb < blocks.numBlocks; lb++)
            {
                if (blocks.blockLevel[lb]==level)
                {
                    Vec3<double> centerCoords = blocks.blockCoords[lb];
                    auto node = domain.Blocks()->GetNodeAt(centerCoords);
                    if (node->GetLevel()<level && (refines.find(node)==refines.end()))
                    {
                        refines.insert({node, 7});
                    }
                }
            }
            if (!refines.empty()) domain.Blocks()->RefineNodes(refines);
        }
        
        std::map<RefinementTreeNode*, int> nodeToBlocks;
        const double tolerance = 0.25*domain.GetMinimumSpacing();
        for (size_t i = 0; i < blocks.numBlocks; i++)
        {
            Vec3<double> blockCenter = blocks.blockCoords[i];
            RefinementTreeNode* nodeAtCenter = domain.Blocks()->GetNodeAt(blockCenter);
            Vec3<double> nodeCenter = nodeAtCenter->GetBlockCenter();
            double err = (blockCenter-nodeCenter).Norm();
            if ((err<tolerance) && domain.ParallelPartitionContainsNode(nodeAtCenter) && (nodeToBlocks.find(nodeAtCenter)==nodeToBlocks.end()))
            {
                nodeToBlocks.insert({nodeAtCenter, i});
            }
        }
        
        auto& arr = domain.DefineVariable("flow", cmf::CmfArrayType::CmfDouble, {5});
        
        size_t numElementsPerBlock = arr.GetArraySizePerBlock();
        
        ParallelFile parFile(domain.GetGroup());
        parFile.Open(flowDataFilename);
        ParallelDataBuffer dataBuf;
        for (auto& p:nodeToBlocks)
        {
            RefinementTreeNode* node = p.first;
            double* rawPointer = (double*)arr[node].pointer;
            dataBuf.Add(rawPointer, numElementsPerBlock, numElementsPerBlock*nodeToBlocks[node]);
        }
        
        parFile.ParallelRead(dataBuf);
        
        return arr;
    }
    
    CartesianMeshInputInfo LegacyRestartReader::ReadMeshInfo(void)
    {
        WriteLine(3, strformat("Reading legacy mesh info from {}", interpolationInfoFilename));
        
        BinaryFile fileHandle(interpolationInfoFilename, fmode::in);
        
        int dim = -1;
        fileHandle >> dim;
        if (dim != CMF_DIM) CmfError(strformat("Error reading legacy restart format: CMF is {}-dimensional, {} is {}-dimensional", CMF_DIM, interpolationInfoFilename, dim));
        std::vector<double> domainBbox;
        domainBbox.resize(2*dim, 0.0);
        for (auto& i:domainBbox) fileHandle >> i;
        int nt;
        fileHandle >> nt;
        int nguard;
        fileHandle >> nguard;
        std::vector<int> nxb;
        nxb.resize(dim, 0);
        for (auto& i:nxb) fileHandle >> i;
        int numBlocksGlob;
        fileHandle >> numBlocksGlob;
        std::vector<double> allBoundingBox;
        allBoundingBox.resize(2*dim*numBlocksGlob);
        for (auto& i:allBoundingBox) fileHandle >> i;
        auto max = [] (double a, double b) -> double {return a<b?b:a;};
        std::vector<double> maxBboxSize;
        maxBboxSize.resize(CMF_DIM, 0.0);
        for (size_t i = 0; i < numBlocksGlob; i++)
        {
            for (int d = 0; d < CMF_DIM; d++)
            {
                double length = allBoundingBox[2*dim*i + 2*d+1] - allBoundingBox[2*dim*i + 2*d];
                maxBboxSize[d] = max(length, maxBboxSize[d]);
            }
        }
        std::vector<int> numInitialBlocks;
        numInitialBlocks.resize(dim, 0);
        for (int d = 0; d < CMF_DIM; d++)
        {
            numInitialBlocks[d] = round((domainBbox[2*d+1]-domainBbox[2*d])/maxBboxSize[d]);
        }
        
        CartesianMeshInputInfo output;
        output.title = "domain";
        output.blockDim = numInitialBlocks;
        output.blockBounds = domainBbox;
        output.periodicRefinement.resize(dim, false);
        output.refinementConstraintType = RefinementConstraint::factor2CompletelyConstrained;
        output.meshDataDim = nxb;
        output.exchangeInfo.interpolationOrder = 2;
        output.exchangeInfo.exchangeDim.resize(dim, nguard);
        return output;
    }
}