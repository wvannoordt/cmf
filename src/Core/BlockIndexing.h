#ifndef CMF_BLOCK_INDEXING_H
#define CMF_BLOCK_INDEXING_H
#include "Config.h"

#if(CMF_IS3D)
#define cmf_kloop(mysymbol, myoffset, myinfo)  for(int mysymbol = -myoffset; mysymbol < myinfo.dataDim[2]+myoffset; mysymbol++)
#define cmf_pkloop(mysymbol, myoffset, myinfo) for(mysymbol = -myoffset; mysymbol < myinfo.dataDim[2]+myoffset; mysymbol++)
#else
#define cmf_kloop(mysymbol, myoffset, myinfo) ;
#define cmf_pkloop(mysymbol, myoffset, myinfo) ;
#endif

#define cmf_jloop(mysymbol, myoffset, myinfo)  for(int mysymbol = -myoffset; mysymbol < myinfo.dataDim[1]+myoffset; mysymbol++)
#define cmf_pjloop(mysymbol, myoffset, myinfo) for(mysymbol = -myoffset; mysymbol < myinfo.dataDim[1]+myoffset; mysymbol++)
#define cmf_iloop(mysymbol, myoffset, myinfo)  for(int mysymbol = -myoffset; mysymbol < myinfo.dataDim[0]+myoffset; mysymbol++)
#define cmf_piloop(mysymbol, myoffset, myinfo) for(mysymbol = -myoffset; mysymbol < myinfo.dataDim[0]+myoffset; mysymbol++)

#if(CMF_IS3D)
#define cmf_idx(myi0, myi1, myi2, myinfo) ((myi0)+myinfo.exchangeDim[0]) + ((myi1)+myinfo.exchangeDim[1])*myinfo.dataDim[0] + ((myi2)+myinfo.exchangeDim[2])*myinfo.dataDim[0]*myinfo.dataDim[1]
#else
#define cmf_idx(myi0, myi1, myi2, myinfo) ((myi0)+myinfo.exchangeDim[0]) + ((myi1)+myinfo.exchangeDim[1])*myinfo.dataDim[0]
#endif

#endif