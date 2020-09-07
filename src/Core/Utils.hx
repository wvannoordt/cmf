#ifndef UTILS_H
#define UTILS_H

#include "Config.h"
#include <cstdlib>
#include <ctime>
#include <iostream>

#define __dloop(mycode) for(int d = 0; d < ANA_DIM; d++) {mycode;}
#define __d_abs(a) (((a)>0)?(a):(-(a)))

static inline void Dim2Idx(int idx, int* dims, int* ijk)
{
    ijk[0] = idx%(dims[0]);
    ijk[1] = ((idx-ijk[0]) / dims[0])%dims[1];
#if(ANA_IS3D)
    ijk[2] = ((idx-ijk[0]-dims[0]*ijk[1])/dims[1]) % dims[2];
#endif
}

static inline int CharBit(char v, int dim)
{
    return 1&(v>>dim);
}

static inline void SetCharBit(char& v, int dim, bool value)
{
    v = (v&(~(1<<dim))) | ((value?7:0)&(1<<dim));
}

static inline int Idx2Dim(int* dims, int* ijk)
{
#if(ANA_IS3D)
    return ijk[0] + ijk[1]*dims[0] + ijk[2]*dims[1]*dims[0];
#else
    return ijk[0] + ijk[1]*dims[0];
#endif
}

static inline int Idx2DimPeriodic(int* dims, int* ijk, char* wasPeriodic)
{
    char output = 0;
    int delta[ANA_DIM];
    for (int d = 0; d < ANA_DIM; d++) delta[d] = 0;
    for (int d = 0; d < ANA_DIM; d++) delta[d] = (ijk[d]<0)?dims[d]:delta[d];
    for (int d = 0; d < ANA_DIM; d++) delta[d] = (ijk[d]>=dims[d])?-dims[d]:delta[d];
    for (int d = 0; d < ANA_DIM; d++) output += (delta[d]==0)?0:(1<<d);
    *wasPeriodic = output;
#if(ANA_IS3D)
    return (ijk[0]+delta[0]) + (ijk[1]+delta[1])*dims[0] + (ijk[2]+delta[2])*dims[1]*dims[0];
#else
    return (ijk[0]+delta[0]) + (ijk[1]+delta[1])*dims[0];
#endif
}

static inline char GetOctant(double* bounds, double* coords)
{
#if(ANA_IS3D)
    return ((2.0*(coords[0]-bounds[0])>=(bounds[1]-bounds[0]))?1:0)
        +  ((2.0*(coords[1]-bounds[2])>=(bounds[3]-bounds[2]))?2:0)
        +  ((2.0*(coords[2]-bounds[4])>=(bounds[5]-bounds[4]))?4:0);
#else
    return ((2.0*(coords[0]-bounds[0])>=(bounds[1]-bounds[0]))?1:0)+((2.0*(coords[1]-bounds[2])>=(bounds[3]-bounds[2]))?2:0);
#endif
}

static inline bool BoxContains(double* bounds, double* coords)
{
#if(ANA_IS3D)
    return (coords[0]>bounds[0])&&(coords[0]<bounds[1])&&(coords[1]>bounds[2])&&(coords[1]<bounds[3])&&(coords[2]>bounds[4])&&(coords[2]<bounds[5]);
#else
    return (coords[0]>bounds[0])&&(coords[0]<bounds[1])&&(coords[1]>bounds[2])&&(coords[1]<bounds[3]);
#endif
}

static inline int RandomInt(int bound)
{
    double r = (double)(rand()%0x000fffff)/((double)0x000fffff);
    return (int)(r*bound)%bound;
}

static inline char BasisEval(int basis, char coord)
{
    return (char)((coord&1)*((basis&0x000000ff)>>0) + ((coord&2)>>1)*((basis&0x0000ff00)>>8) + ((coord&4)>>2)*((basis&0x00ff0000)>>16));
}

static inline char BasisEvalReverse(int basis, char coord)
{
    return (char)((coord&1)*((basis&0x00ff0000)>>16) + ((coord&2)>>1)*((basis&0x0000ff00)>>8) + ((coord&4)>>2)*((basis&0x000000ff)>>0));
}


static inline void PrintIntAsVector(int p)
{
    std::cout << ((p&0xff000000)>>24) << ", ";
    std::cout << ((p&0x00ff0000)>>16) << ", ";
    std::cout << ((p&0x0000ff00)>>8 ) << ", ";
    std::cout << ((p&0x000000ff)>>0 ) << std::endl;
}

static inline void PrintCharAsVector(char p)
{
    std::cout << ((p&4)>>2) << ", ";
    std::cout << ((p&2)>>1) << ", ";
    std::cout << ((p&1)>>0) << std::endl;
}

#endif
