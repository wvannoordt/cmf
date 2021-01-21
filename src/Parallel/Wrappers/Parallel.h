#ifndef CMF_PARRALEL_H
#define CMF_PARRALEL_H
#include "CmfMPI.h"
namespace cmf
{
    /// @brief Initializes MPI (if enabled) and sets the global parallel group communicator to the default MPI one
    /// @param argc Pointer to argc from main()
    /// @param argv Pointer to argv from main()
    /// @author WVN
    void CreateParallelContext(int* argc, char*** argv);

    /// @brief Indicates if MPI_Finalize() should be called by the main parallel group
    extern bool handleMpiFinalizationInternally;
}

#endif