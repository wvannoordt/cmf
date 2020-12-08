#include "Parallel.h"
#include "ParallelGroup.h"
#include "CmfError.h"
#include "CmfScreen.h"
namespace cmf
{
    bool handleMpiFinalizationInternally = false;
    void CreateParallelContext(int* argc, char*** argv)
    {
#if(CMF_PARALLEL)
        int MPIIsInitialized = 0;
        CMF_MPI_CHECK(MPI_Initialized(&MPIIsInitialized));
        if (!MPIIsInitialized)
        {
            handleMpiFinalizationInternally = true;
            if (!argc)
            {
                WriteLine(1, "WARNING: MPI has been initialized by the with no arguments");
            }
            CMF_MPI_CHECK(MPI_Init(argc,argv));
        }
        globalGroup.CreateGroup();
        if (handleMpiFinalizationInternally)
        {
            WriteLine(2, "Initialized MPI");
        }
#else
        CmfError("Attempted to call CreateParallelContext, but CMF was compiled without parallel support!");
        handleMpiFinalizationInternally = true;
#endif
    }
}