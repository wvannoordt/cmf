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
            
            /// @brief Returns the process ID of the current process
            /// @author WVN
            int Rank(void);
            
            /// @brief Returns the size of the current communicator world
            /// @author WVN
            int Size(void);
            
            /// @brief Waits for all processes, similar to MPI_Barrier.
            /// @author WVN
            void Synchronize(void);
        
        private:
            
            /// @brief Rank of the current process
            int processId;
            
            /// @brief Number of processes in this parallel group
            int processCount;
            
            /// @brief Indicates whether the current process is the root process
            bool isRoot;
            
            /// @brief The underlying communicator object
            CmfParallelCommunicator communicator;        
    };

    /// @brief The default parallel group for global parallel operations
    extern ParallelGroup globalGroup;
}

#endif