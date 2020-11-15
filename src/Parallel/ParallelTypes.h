#ifndef CMF_PARALLEL_TYPES
#define CMF_PARALLEL_TYPES
#include "CmfMPI.h"

namespace cmf
{
#if(CMF_PARALLEL)
    typedef MPI_Comm CmfParallelCommunicator;
#else
    typedef unsigned long CmfParallelCommunicator;
#endif

    /// @brief The default communicator (MPI_COMM_WORLD)
    extern CmfParallelCommunicator defaultCommunicator;
}
#endif