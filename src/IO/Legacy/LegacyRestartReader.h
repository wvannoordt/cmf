#ifndef CMF_LEGACY_RESTART_READER_H
#define CMF_LEGACY_RESTART_READER_H
#include <string>
#include "CartesianMesh.h"
#include "CmfError.h"
#include "StringUtils.h"
#include "Vec.h"
namespace cmf
{
    /// @brief A holder class for the legacy block format
    /// @author WVN
    struct LegacyRestartBlockArrangement
    {
        public:
            /// @brief Constructor, reads the provided file
            /// @param filename The file to read
            /// @author WVN
            LegacyRestartBlockArrangement(std::string filename)
            {
                {
                    std::ifstream t(filename);
                    std::stringstream buffer;
                    buffer << t.rdbuf();
                    buffer >> this->numBlocks;
                    buffer >> this->maxLevel;
                    size_t numNumericEntries = 0;
                    double dummy;
                    while (!buffer.fail())
                    {
                        buffer >> dummy;
                        numNumericEntries++;
                    }
                    numNumericEntries--;
                    this->dim = numNumericEntries/(this->numBlocks) - 3;
                    if (this->dim!=2 && this->dim!=3) CmfError(strformat("Couldn't parse file \"{}\" as legacy block format: bad dimension ({})", filename, dim));
                }
                blockLevel.resize(this->numBlocks);
                blockCoords.resize(this->numBlocks);
                {
                    std::ifstream t(filename);
                    std::stringstream buffer;
                    buffer << t.rdbuf();
                    buffer >> this->numBlocks;
                    buffer >> this->maxLevel;
                    int dummy;
                    for (size_t lb = 0; lb < this->numBlocks; lb++)
                    {
                        auto& xyz = blockCoords[lb];
                        xyz[0] = 0.0; xyz[1] = 0.0; xyz[2] = 0.0;
                        buffer >> dummy;
                        buffer >> blockLevel[lb];
                        blockLevel[lb]--;
                        buffer >> dummy;
                        for (int d = 0; d < this->dim; d++)
                        {
                            buffer >> xyz[d];
                        }
                    }
                }
                maxLevel = -1;
                auto max = [](int a, int b) -> int {return a<b?b:a;};
                for (auto& i: blockLevel) maxLevel = max(maxLevel,i);
            }
            
            ///@brief the number of blocks in total
            int numBlocks;
            
            ///@brief max refinement level
            int maxLevel;
            
            ///@brief dimension of the data
            int dim;
            
            ///@brief refinement level of each block
            std::vector<int> blockLevel;
            
            ///@brief center of each block
            std::vector<Vec3<double>> blockCoords;
            
    };
    /// @brief A reader class for the legacy restart file format
    /// @author WVN
    class LegacyRestartReader
    {
        public:
            /// @brief Constructor
            /// @param interpolationInfo_in File name for the gridInterpolationInfo_... file
            /// @param blockInfo_in File name for the restart_block_arrangement_nt_... file
            /// @author WVN
            LegacyRestartReader(std::string interpolationInfo_in, std::string blockInfo_in);
            
            /// @brief Returns a CartesianMeshInputInfo that can be used to create a CartesianMesh
            /// @author WVN
            CartesianMeshInputInfo ReadMeshInfo(void);
            
            /// @brief Refines the provided mesh to conform to the written data, then loads the flowfield data to the mesh
            /// @param domain the mesh to apply refinement to/load the data to
            /// @param flowData File name for the restart_unk_nt_... file
            /// @author WVN
            CartesianMeshArray& LoadToMesh(CartesianMesh& domain, std::string flowData);
        private:
            /// @brief File name for the gridInterpolationInfo_... file
            std::string interpolationInfoFilename;
            
            /// @brief File name for the restart_block_arrangement_nt_... file
            std::string blockInfoFilename;
    };
}

#endif