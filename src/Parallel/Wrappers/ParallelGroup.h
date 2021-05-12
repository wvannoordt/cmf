#ifndef CMF_PARALLEL_GROUP
#define CMF_PARALLEL_GROUP
#include "CmfMPI.h"
#include "Parallel.h"
#include "ParallelTypes.h"
#include "CudaDeviceHandler.h"
#include "ICmfHashable.h"
#include "CmfPrint.h"
namespace cmf
{
    /// @brief Class defining a parrallel group around a parallel communicator.
	/// @author WVN
    class ParallelGroup : ICmfHashable
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
            template <typename sType> sType* SharedValues(sType n)
            {
                char* buf = (char*)(&n);
                int ct = (int)sizeof(sType);
                AllGather(buf, ct, parallelChar, workArray, ct, parallelChar);
                return (sType*)workArray;
            }
            
            /// @brief Returns the sum of the values on all ranks
            /// @param val The value to sum
            /// @author WVN
            size_t Sum(size_t val);
            
            /// @brief Returns the sum of the values on all ranks
            /// @param val The value to sum
            /// @author WVN
            double Sum(double val);
            
            /// @brief Returns the max of the values on all ranks
            /// @param val The value to sum
            /// @author WVN
            double Max(double val);
            
            /// @brief Returns the sum of the values on all ranks
            /// @param val The value to sum
            /// @author WVN
            int Sum(int val);
            
            /// @brief Returns the parallel communicator
            /// @author WVN
            CmfParallelCommunicator GetComm(void) { return communicator; }
            
            /// @brief Eqivalent to <a href="https://www.mpich.org/static/docs/v3.3/www3/MPI_Irecv.html">MPI_Irecv</a>
            /// @param buf initial address of receive buffer
            /// @param count number of elements in receive buffer
            /// @param datatype datatype of each receive buffer element
            /// @param source rank of source
            /// @param request communication request (output)
            void QueueReceive(void *buf, int count, ParallelDataType datatype, int source, ParallelRequestHandle* request);
            
            /// @brief Eqivalent to <a href="https://www.mpich.org/static/docs/v3.3/www3/MPI_Ssend.html">MPI_Ssend</a>
            /// @param buf initial address of send buffer
            /// @param count number of elements in receive buffer
            /// @param datatype datatype of each send buffer element
            /// @param dest rank of destination
            void BlockingSynchronousSend(const void *buf, int count, ParallelDataType datatype, int dest);
            
            /// @brief Eqivalent to <a href="https://www.mpich.org/static/docs/v3.3/www3/MPI_Waitall.html">MPI_Waitall</a>
            /// @param count list length
            /// @param arrayOfRequests array of request handles
            /// @param arrayOfStatuses array of status objects (output)
            void AwaitAllAsynchronousOperations(int count, ParallelRequestHandle arrayOfRequests[], ParallelStatus arrayOfStatuses[]);
            
            /// @brief Eqivalent to <a href="https://www.mpich.org/static/docs/v3.2/www3/MPI_Allgather.html">MPI_Allgather</a>
            /// @param sendbuf starting address of send buffer
            /// @param sendcount number of elements in send buffer
            /// @param sendtype data type of send buffer elements
            /// @param recvbuf starting address of received buffer
            /// @param recvcount number of elements received from any process
            /// @param recvtype data type of receive buffer elements
            /// @author WVN
            void AllGather(const void *sendbuf, int sendcount, ParallelDataType sendtype, void *recvbuf, int recvcount, ParallelDataType recvtype);
            
            /// @brief Eqivalent to <a href="https://www.mpich.org/static/docs/v3.2/www3/MPI_Gatherv.html">MPI_Gatherv</a>
            /// @param sendbuf starting address of send buffer
            /// @param sendcount number of elements in send buffer
            /// @param sendtype data type of send buffer elements
            /// @param recvbuf starting address of received buffer
            /// @param recvcounts number of elements received from any process
            /// @param displs integer array (of length group size). Entry i specifies the displacement (relative to recvbuf) at which to place the incoming data from process i
            /// @param recvtype data type of receive buffer elements
            /// @author WVN
            void Gatherv(const void *sendbuf, int sendcount, ParallelDataType sendtype, void *recvbuf, const int* recvcounts, const int* displs, ParallelDataType recvtype);
            
            /// @brief Eqivalent to <a href="https://www.mpich.org/static/docs/latest/www3/MPI_Allreduce.html">MPI_Allreduce</a>
            /// @param sendbuf starting address of send buffer
            /// @param recvbuf starting address of received buffer
            /// @param count number of elements received from any process
            /// @param datatype data type of send buffer elements
            /// @param op parallel operation
            /// @author WVN
            void AllReduce(const void *sendbuf, void *recvbuf, int count, ParallelDataType datatype, ParallelOperation op);
            
            /// @brief Returns a vector on only the root rank that contains the rank-ordered and appended list provided.
            ///Should not be used for large amounts of data
            /// @param inputList the elements on the local rank
            /// @param outputList the concatenated elements on the root ranks (output, this vector will be cleared)
            /// @author WVN
            template <typename gtype> void GetAppendedListOnRoot(std::vector<gtype>& inputList, std::vector<gtype>& outputList)
            {
                outputList.clear();
                size_t globNumElems = this->Sum(inputList.size());
                std::vector<int> displacements;
                displacements.resize(this->Size(), 0);
                int* numberElementsPerRank = this->SharedValues((int)inputList.size());
                displacements[0] = 0;
                for (int i = 1; i < this->Size(); i++)
                {
                    displacements[i] = displacements[i-1] + numberElementsPerRank[i-1];
                }
                if (this->IsRoot()) outputList.resize(globNumElems);
                Gatherv(&inputList[0], inputList.size(), parallelInt, &outputList[0], numberElementsPerRank, &displacements[0], parallelInt);
            }
            /// @brief Initializes MPI if required
            /// @author WVN
            void MpiAutoInitIfRequired(void);
            
            /// @brief Returns isRoot
            /// @author WVN
            bool IsRoot(void);
        
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
            
            /// @brief Indicates whether the parallel group is operating in serial mode
            bool serialMode;
            
            /// @brief Indicates whether or not MPI was initialized on the first MPI call
            bool autoInitialized;
            
            /// @brief Indicates whether or not MpiAutoInitIfRequired has been called
            bool mpiAutoInitIfRequiredCalled;
            
            /// @brief A handler object for CUDA gpu devices attached to the current node
            CudaDeviceHandler* deviceHandler;
            
            /// @brief Indicates whether or not this parallelgroup needs to delete the deviceHandler
            bool deleteCudaDeviceHandler;
            
            /// @brief Counts the number of times Synchronize() is called, used for debugging
            int synchCount;
            
            /// @brief The rank of the root rank
            int rootRank;
    };

    /// @brief The default parallel group for global parallel operations
    extern ParallelGroup globalGroup;
}

#endif