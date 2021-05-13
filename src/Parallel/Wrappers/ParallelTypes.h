#ifndef CMF_PARALLEL_TYPES
#define CMF_PARALLEL_TYPES
#include "CmfMPI.h"
#include <stdio.h>
#include "CmfError.h"
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
    
#if(CMF_PARALLEL)
    typedef MPI_File   CmfMpiFileHandle;
    typedef MPI_Status CmfMpiFileStatus;
#else
    typedef int        CmfMpiFileHandle;
    typedef int        CmfMpiFileStatus;
#endif

    template <typename ptype> ParallelDataType GetParallelType(void)
    {
        if (typeid(ptype) == typeid(int))    return parallelInt;
        if (typeid(ptype) == typeid(double)) return parallelDouble;
        if (typeid(ptype) == typeid(size_t)) return parallelLong;
        if (typeid(ptype) == typeid(char))   return parallelChar;
        CmfError("Requested invalid type, no further information is available here, unfortunately");
        return 0;
    }
    
}
#endif