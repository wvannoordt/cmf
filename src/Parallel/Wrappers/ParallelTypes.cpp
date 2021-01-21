#include "ParallelTypes.h"

namespace cmf
{
    
#if(CMF_PARALLEL)
    CmfParallelCommunicator defaultCommunicator = MPI_COMM_WORLD;
    ParallelDataType parallelDouble = MPI_DOUBLE;
    ParallelDataType parallelInt = MPI_INT;
    ParallelDataType parallelLong = MPI_LONG;
    ParallelOperation parallelSum = MPI_SUM;
#else
    CmfParallelCommunicator defaultCommunicator = 0;
    ParallelDataType parallelDouble = 0;
    ParallelDataType parallelInt = 0;
    ParallelDataType parallelLong = 0;
    ParallelOperation parallelSum = 0;
#endif

}