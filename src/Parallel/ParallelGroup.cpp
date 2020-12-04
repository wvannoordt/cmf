#include "ParallelGroup.h"
#include "CmfScreen.h"
#include "CmfGC.h"
#include "cmf.h"
namespace cmf
{
    ParallelGroup globalGroup;
    
    ParallelGroup::ParallelGroup(void)
    {
        processId = 0;
        processCount = 1;
        communicator = defaultCommunicator;
        workArray = NULL;
        deallocWorkArray = false;
        isInitialized = false;
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
        deallocWorkArray = true;
        SetStackAllocationAllowed(false);
        workArray = Cmf_Alloc(processCount*sizeof(size_t));
        SetStackAllocationAllowed(true);
        isInitialized = true;
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
        if (deallocWorkArray)
        {
            Cmf_Free(workArray);
            deallocWorkArray = false;
        }
    }
    
    void ParallelGroup::Synchronize(void)
    {
        CMF_MPI_CHECK(MPI_Barrier(communicator));
    }
    
    bool ParallelGroup::HasSameValue(int n)
    {
        if (!isInitialized) return true;
        int* sharedArray = (int*)SharedValues(n);
        bool output = true;
        for (int i = 0; i < processCount; i++) output = output && (sharedArray[i] == n);
        return output;
    }
    
    void* ParallelGroup::SharedValues(int n)
    {
        AllGather(&n, 1, parallelInt, workArray, 1, parallelInt);
        return workArray;
    }
    
    void ParallelGroup::AllGather(const void *sendbuf, int sendcount, ParallelDataType sendtype, void *recvbuf, int recvcount, ParallelDataType recvtype)
    {
        CMF_MPI_CHECK(MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, communicator));
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