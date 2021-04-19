#ifndef CMF_MPI_H
#define CMF_MPI_H
#include "ConfigParams.h"
#ifndef CMF_PARALLEL
#define CMF_PARALLEL 0
#endif

#include "CmfError.h"
#if(CMF_PARALLEL)
#include "mpi.h"
#define CMF_MPI_CHECK(myMpiCall) if (myMpiCall != MPI_SUCCESS) {std::string er______MACRO(#myMpiCall); CmfError("MPI threw error from the following call: \"" + er______MACRO + "\".");}
#else
#define CMF_MPI_CHECK(myMpiCall) ;
#endif

#endif