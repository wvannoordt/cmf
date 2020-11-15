#include "ParallelTypes.h"

namespace cmf
{
    
#if(CMF_PARALLEL)
    CmfParallelCommunicator defaultCommunicator = MPI_COMM_WORLD;
#else
    CmfParallelCommunicator defaultCommunicator = 0;
#endif

}