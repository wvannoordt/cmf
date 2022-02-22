#ifndef CMF_H
#define CMF_H
#include "CmfError.h"
#include <string>
#include "Config.h"
#include "PTL.h"
#include "CartesianMesh.h"
#include "RefinementBlock.h"
#include "RefinementTreeNode.h"
#include "NeighborIterator.h"
#include "VtkFile.h"
#include "PngImage.h"
#include "PngDrawer.h"
#include "BlockFilters.h"
#include "AxisAlignedLongCylinder.h"
#include "CmfGC.h"
#include "CmfScreen.h"
#include "SurfaceTriangulation.h"
#include "GlobalSettings.h"
#include "CartesianMeshArray.h"
#include "SerialCartesianVtk.h"
#include "Parallel.h"
#include "ParallelGroup.h"
#include "BlockIndexing.h"
#include "BlockVtk.h"
#include "SvgImage.h"
#include "DataExchangePattern.h"
#include "Vec.h"
#include "BlockArray.h"
#include "CmfPrint.h"
#include "StringUtils.h"
#include "CmfDataBase.h"
#include "CmfGlobalVariables.h"
#include "LegacyRestartReader.h"
#include "CartesianMeshInterpolationOperator.h"

namespace cmf
{

    /// @brief Initializes CMF. Currently does nothing. It is intended that this function does nothing.
    /// @author WVN
    void Initialize(void);
    
    /// @brief Reads the input file (PTL format) specified by filename. PTL format is specific to <a href="https://github.com/wvannoordt/PTL">PTL</a>.
    /// @param filename The name if the input file to be read
    /// @author WVN
    void ReadInput(std::string filename);
    
    /// @brief Finalizes CMF. Currently does nothing. It is intended that this function does nothing.
    /// @author WVN
    void Finalize(void);
    
    /// @brief Returns CMF_DIM, used to check that the host code is using the correct dimension (determined at compile time) of CMF.\see Config.h
    /// @author WVN
    constexpr int Dim(void)
    {
        return CMF_DIM;
    }
    
    /// @brief Returns CMF_DIM, used to check that the host code is using the correct dimension (determined at compile time) of CMF.\see Config.h
    /// @author WVN
    constexpr int GetDim(void)
    {
        return CMF_DIM;
    }
    
    /// @brief Returns true if CMF was configured in 3D
    /// @author WVN
    constexpr bool Is3D(void)
    {
        return CMF_DIM==3;
    }
    
    /// @brief Returns true if CMF was compiled with parallel MPI support
    /// @author WVN
    constexpr bool IsParallel(void)
    {
        return CMF_PARALLEL?true:false;
    }
    
    /// @brief Returns true if CMF was compiled with GPU support
    /// @author WVN
    constexpr bool HasGpuSupport(void)
    {
        return CUDA_ENABLE?true:false;
    }
}

#endif
