#include "ParallelTypes.h"

namespace cmf
{
    
#if(CMF_PARALLEL)
    CmfParallelCommunicator defaultCommunicator = MPI_COMM_WORLD;
    ParallelDataType parallelDouble = MPI_DOUBLE;
    ParallelDataType parallelInt = MPI_INT;
    ParallelDataType parallelChar = MPI_CHAR;
    ParallelDataType parallelLong = MPI_LONG;
    ParallelOperation parallelSum = MPI_SUM;
    ParallelOperation parallelMax = MPI_MAX;
#else
    CmfParallelCommunicator defaultCommunicator = 0;
    ParallelDataType parallelDouble = 0;
    ParallelDataType parallelInt = 0;
    ParallelDataType parallelChar = 0;
    ParallelDataType parallelLong = 0;
    ParallelOperation parallelSum = 0;
    ParallelOperation parallelMax = 0;
#endif

}