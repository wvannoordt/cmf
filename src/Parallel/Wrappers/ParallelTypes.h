#ifndef CMF_PARALLEL_TYPES
#define CMF_PARALLEL_TYPES
#include "CmfMPI.h"

namespace cmf
{
#if(CMF_PARALLEL)
    typedef MPI_Comm      CmfParallelCommunicator;
    typedef MPI_Datatype  ParallelDataType;
    typedef MPI_Op        ParallelOperation;
#else
    typedef unsigned long CmfParallelCommunicator;
    typedef unsigned long ParallelDataType;
    typedef unsigned long ParallelOperation;
#endif

    /// @brief The default communicator (MPI_COMM_WORLD)
    extern CmfParallelCommunicator defaultCommunicator;
    
    
    /// @brief double type
    extern ParallelDataType parallelDouble;
    
    /// @brief int type
    extern ParallelDataType parallelInt;
    
    /// @brief long type
    extern ParallelDataType parallelLong;
    
    /// @brief parallel sum operation
    extern ParallelOperation parallelSum;
    
}
#endif