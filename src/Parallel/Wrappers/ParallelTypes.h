#ifndef CMF_PARALLEL_TYPES
#define CMF_PARALLEL_TYPES
#include "CmfMPI.h"

namespace cmf
{
#if(CMF_PARALLEL)
    typedef MPI_Comm      CmfParallelCommunicator;
    typedef MPI_Datatype  ParallelDataType;
    typedef MPI_Op        ParallelOperation;
    typedef MPI_Request   ParallelRequestHandle;
    typedef MPI_Status    ParallelStatus;
#else
    typedef unsigned long CmfParallelCommunicator;
    typedef unsigned long ParallelDataType;
    typedef unsigned long ParallelOperation;
    typedef unsigned long ParallelRequestHandle;
    typedef unsigned long ParallelStatus;
#endif

    /// @brief The default communicator (MPI_COMM_WORLD)
    extern CmfParallelCommunicator defaultCommunicator;
    
    
    /// @brief double type
    extern ParallelDataType parallelDouble;
    
    /// @brief int type
    extern ParallelDataType parallelInt;
    
    /// @brief int type
    extern ParallelDataType parallelChar;
    
    /// @brief long type
    extern ParallelDataType parallelLong;
    
    /// @brief parallel sum operation
    extern ParallelOperation parallelSum;
    
    /// @brief parallel sum operation
    extern ParallelOperation parallelMax;
    
}
#endif