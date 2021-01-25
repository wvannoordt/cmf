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
        serialMode = true;
        mpiAutoInitIfRequiredCalled = false;
        deleteCudaDeviceHandler = false;
    }
    
    
    void ParallelGroup::CreateGroup(void)
    {
        CreateGroup(defaultCommunicator);
        if (CUDA_ENABLE)
        {
            deleteCudaDeviceHandler = true;
            deviceHandler = new CudaDeviceHandler();
        }
    }
    
    void ParallelGroup::CreateGroup(CmfParallelCommunicator comm)
    {
        communicator = comm;
        CMF_MPI_CHECK(MPI_Comm_rank(comm, &processId));
        CMF_MPI_CHECK(MPI_Comm_size(comm, &processCount));
        isRoot = (processId==0);
        serialMode = false;
        globalSettings.globalOutputEnabledHere = isRoot;
        deallocWorkArray = true;
        SetStackAllocationAllowed(false);
        workArray = Cmf_Alloc(processCount*sizeof(size_t));
        SetStackAllocationAllowed(true);
        isInitialized = true;
        if (!CMF_PARALLEL) serialMode = true;
    }
    
    ParallelGroup::~ParallelGroup(void)
    {
        if (deleteCudaDeviceHandler)
        {
            deleteCudaDeviceHandler = false;
            delete deviceHandler;
        }
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
    
    void ParallelGroup::QueueReceive(void *buf, int count, ParallelDataType datatype, int source, ParallelRequestHandle* request)
    {
        CMF_MPI_CHECK(MPI_Irecv(buf, count, datatype, source, 1, communicator, request));
#if(!CMF_PARALLEL)
        CmfError("ParallelGroup::QueueReceive called without parallel support. This will probably cause an issue. Contact WVN");
#endif
    }
    
    void ParallelGroup::BlockingSynchronousSend(const void *buf, int count, ParallelDataType datatype, int dest)
    {
        CMF_MPI_CHECK(MPI_Ssend(buf, count, datatype, dest, 1, communicator));
#if(!CMF_PARALLEL)
        CmfError("ParallelGroup::BlockingSynchronousSend called without parallel support. This will probably cause an issue. Contact WVN");
#endif
    }

    void ParallelGroup::AwaitAllAsynchronousOperations(int count, ParallelRequestHandle arrayOfRequests[], ParallelStatus arrayOfStatuses[])
    {
        CMF_MPI_CHECK(MPI_Waitall(count, arrayOfRequests, arrayOfStatuses));
#if(!CMF_PARALLEL)
        CmfError("ParallelGroup::AwaitAllAsynchronousOperations called without parallel support. This will probably cause an issue. Contact WVN");
#endif
    }
    
    bool ParallelGroup::IsInitialized(void)
    {
        return isInitialized;
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
        MpiAutoInitIfRequired();
        CMF_MPI_CHECK(MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, communicator));
    }
    
    void ParallelGroup::AllReduce(const void *sendbuf, void *recvbuf, int count, ParallelDataType datatype, ParallelOperation op)
    {
        MpiAutoInitIfRequired();
        CMF_MPI_CHECK(MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, communicator));
    }
    
    void ParallelGroup::MpiAutoInitIfRequired(void)
    {
        if (mpiAutoInitIfRequiredCalled) return;
        mpiAutoInitIfRequiredCalled = true;
        int MPIIsInitialized = 1;
        CMF_MPI_CHECK(MPI_Initialized(&MPIIsInitialized));
        if (!MPIIsInitialized)
        {
            CMF_MPI_CHECK(MPI_Init(NULL,NULL));
            autoInitialized = true;
            handleMpiFinalizationInternally = true;
            CreateGroup();
            if (Size()==1)
            {
                WriteLine(1, "WARNING: MPI was initialized automatically by a ParallelGroup call. It is recommended to use cmf::CreateParallelContext instead");
            }
        }
    }
    
    bool ParallelGroup::IsRoot(void)
    {
        return isRoot;
    }
    
    size_t ParallelGroup::Sum(size_t val)
    {
        if (serialMode) return val;
        size_t gVal = 0;
        AllReduce(&val, &gVal, 1, parallelLong, parallelSum);
        return gVal;
    }
    
    double ParallelGroup::Sum(double val)
    {
        if (serialMode) return val;
        double gVal = 0;
        AllReduce(&val, &gVal, 1, parallelDouble, parallelSum);
        return gVal;
    }
    
    int ParallelGroup::Sum(int val)
    {
        if (serialMode) return val;
        int gVal = 0;
        AllReduce(&val, &gVal, 1, parallelInt, parallelSum);
        return gVal;
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