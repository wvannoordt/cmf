#ifndef DEBUG_TOOLS_H
#define DEBUG_TOOLS_H
#include "HybridComputing.h"
#include "CmfOutputStream.h"
#include <iostream>

#define __longline "---------------------------------------------------------"

#if __cmf_cpu
#define __dump(mystuff) {cmf::cmfout << __FILE__ <<  " (" << __LINE__ << "): " << mystuff << cmf::cmfendl;}
#define __qdump(mystuff) {cmf::cmfout << mystuff << cmf::cmfendl;}
#define __sdump(mystuff) {cmf::cmfout << #mystuff << ": " <<  mystuff << cmf::cmfendl;}
#define __erkill(mystuff) {cmf::cmfout << __longline <<std::endl << "Terminate called from " << __FILE__ <<  ", line (" << __LINE__ << "): " << mystuff << std::endl << __longline << cmf::cmfendl; abort();}
#endif


#if __cmf_gpu
#define __dump(mystuff) ;
#define __qdump(mystuff) ;
#define __sdump(mystuff) ;
#define __dumpwait(mystuff) ;
#define __erkill(mystuff) ;
#endif

#endif
