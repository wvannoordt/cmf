#include "ParallelGroup.h"
#include "CmfScreen.h"
#include "cmf.h"
namespace cmf
{
    ParallelGroup globalGroup;
    
    ParallelGroup::ParallelGroup(void)
    {
        processId = 0;
        processCount = 1;
        communicator = defaultCommunicator;
    }
    
    
    void ParallelGroup::CreateGroup(void)
    {
        CreateGroup(defaultCommunicator);
    }
    
    void ParallelGroup::CreateGroup(CmfParallelCommunicator comm)
    {
        communicator = comm;
        CMF_MPI_CHECK(MPI_Comm_rank(comm, &processId));
        CMF_MPI_CHECK(MPI_Comm_size(comm, &processCount));
        isRoot = (processId==0);
        globalSettings.globalOutputEnabledHere = isRoot;
    }
    
    ParallelGroup::~ParallelGroup(void)
    {
        if (handleMpiFinalizationInternally)
        {
            int flag;
            CMF_MPI_CHECK(MPI_Finalized(&flag));
            if (!flag)
            {
                WriteLine(2, "Finalize MPI");
                CMF_MPI_CHECK(MPI_Finalize());
            }
        }
    }
    
    void ParallelGroup::Synchronize(void)
    {
        CMF_MPI_CHECK(MPI_Barrier(communicator));
    }
    
    int ParallelGroup::Rank(void)
    {
        return processId;
    }
    
    int ParallelGroup::Size(void)
    {
        return processCount;
    }
}