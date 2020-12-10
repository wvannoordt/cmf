#ifndef CMF_PARALLEL_GROUP
#define CMF_PARALLEL_GROUP
#include "CmfMPI.h"
#include "Parallel.h"
#include "ParallelTypes.h"
namespace cmf
{
    /// @brief Class defining a parrallel group around a parallel communicator.
	/// @author WVN
    class ParallelGroup
    {
        public:
            
            /// @brief Default constructor for the parallel group, currently defaults to the global communicator
        	/// @author WVN
            ParallelGroup(void);
            
            /// @brief Default destructor
            /// @author WVN
            ~ParallelGroup(void);
            
            /// @brief Sets the communicator and communicates ranks
            /// @author WVN
            void CreateGroup(void);
            
            /// @brief Sets the communicator and communicates ranks
            /// @param comm The communicator to pass to the parallel group
            /// @author WVN
            void CreateGroup(CmfParallelCommunicator comm);
            
            /// @brief Returns isInitialized
            /// @author WVN
            bool IsInitialized(void);
            
            /// @brief Returns the process ID of the current process
            /// @author WVN
            int Rank(void);
            
            /// @brief Returns the size of the current communicator world
            /// @author WVN
            int Size(void);
            
            /// @brief Waits for all processes, similar to MPI_Barrier.
            /// @author WVN
            void Synchronize(void);
            
            /// @brief Indicates whether or not this parallel pool agrees on the given value
            /// @param n The value to check
            /// @author WVN
            bool HasSameValue(int n);
            
            /// @brief Returns an array with the value given by process i stored at location i \pre Note that the returned array is only valid until the next call of this function!
            /// @param n The value to share
            /// @author WVN
            void* SharedValues(int n);
            
            /// @brief Eqivalent to <a href="https://www.mpich.org/static/docs/v3.2/www3/MPI_Allgather.html">MPI_Allgather</a>
            /// @param sendbuf starting address of send buffer
            /// @param sendcount number of elements in send buffer
            /// @param sendtype data type of send buffer elements
            /// @param recvbuf starting address of received buffer
            /// @param recvcount number of elements received from any process
            /// @param recvtype data type of receive buffer elements
            /// @author WVN
            void AllGather(const void *sendbuf, int sendcount, ParallelDataType sendtype, void *recvbuf, int recvcount, ParallelDataType recvtype);
            
            /// @brief Eqivalent to <a href="https://www.mpich.org/static/docs/latest/www3/MPI_Allreduce.html">MPI_Allreduce</a>
            /// @param sendbuf starting address of send buffer
            /// @param recvbuf starting address of received buffer
            /// @param count number of elements received from any process
            /// @param datatype data type of send buffer elements
            /// @param op parallel operation
            /// @author WVN
            void AllReduce(const void *sendbuf, void *recvbuf, int count, ParallelDataType datatype, ParallelOperation op);
        
        private:
            
            /// @brief Rank of the current process
            int processId;
            
            /// @brief Number of processes in this parallel group
            int processCount;
            
            /// @brief Indicates whether the current process is the root process
            bool isRoot;
            
            /// @brief The underlying communicator object
            CmfParallelCommunicator communicator;
            
            /// @brief An array used as a work array for simple exchanges
            void* workArray;
            
            /// @brief Indicates whether workArray requires freeing
            bool deallocWorkArray;
            
            /// @brief Indicates the parallel group has been created using CreateGroup
            bool isInitialized;
    };

    /// @brief The default parallel group for global parallel operations
    extern ParallelGroup globalGroup;
}

#endif