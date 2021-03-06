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

namespace cmf
{
    extern PTL::PropertyTree mainInput;
    extern GlobalSettings globalSettings;
    
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
    int GetDim(void);
    
    /// @brief Returns true if CMF was compiled with parallel MPI support
    /// @author WVN
    bool IsParallel(void);
    
    /// @brief Returns true if CMF was compiled with GPU support
    /// @author WVN
    bool HasGpuSupport(void);
}

#endif
