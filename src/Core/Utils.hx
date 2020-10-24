#ifndef UTILS_H
#define UTILS_H

#include "Config.h"
#include <cstdlib>
#include <ctime>
#include <iostream>

/// \file
///
/// Provides some simple utilities, almost all of which have different behavior depending on CMF_DIM. \see Config.h
///
/// @author WVN

#define __dloop(mycode) for(int d = 0; d < CMF_DIM; d++) {mycode;}
#define __d_abs(a) (((a)>0)?(a):(-(a)))
#define CMFMAX(a,b) ((a)>(b)?(a):(b))
#define CMFMIN(a,b) ((a)<(b)?(a):(b))

/// @brief Transforms 1-D index into 2-D or 3-D index. For example, if idx is 3, CMF_DIM is 2, and dims = (2, 2), then ijk = (0, 1) \see Idx2Dim
/// @pre size of dims and ijk must be equal to CMF_DIM \see Config.h
/// @param idx 1-D index
/// @param dims 2-D or 3-D size
/// @param ijk  (output) resulting 2-D or 3-D index
/// @author WVN
static inline void Dim2Idx(int idx, int* dims, int* ijk)
{
    ijk[0] = idx%(dims[0]);
    ijk[1] = ((idx-ijk[0]) / dims[0])%dims[1];
#if(CMF_IS3D)
    ijk[2] = ((idx-ijk[0]-dims[0]*ijk[1])/(dims[1]*dims[0])) % dims[2];
#endif
}

/// @brief Gets the bit (as an integer) in a position of a char.
/// @param v a char
/// @param dim the index of the bit to be retrieved
/// @author WVN
static inline int CharBit(char v, int dim)
{
    return 1&(v>>dim);
}

/// @brief Sets the bit in a position of a char.
/// @param v a char
/// @param dim the index of the bit to be set
/// @param value the value of the bit to be set
/// @author WVN
static inline void SetCharBit(char& v, int dim, bool value)
{
    v = (v&(~(1<<dim))) | ((value?7:0)&(1<<dim));
}

/// @brief Returns a 1-D index based on a 2-D or 3-D array. ijk[0] is the minor index, followed by ijk[1], and, if compiled with CMF_DIM=3, then ijk[2]. The inverse of Dim2Idx. \see Dim2Idx
/// @param dims a 2-D or 3-D array of dimensions
/// @param ijk a 2-D or 3-D array of indices
/// @author WVN
static inline int Idx2Dim(int* dims, int* ijk)
{
#if(CMF_IS3D)
    return ijk[0] + ijk[1]*dims[0] + ijk[2]*dims[1]*dims[0];
#else
    return ijk[0] + ijk[1]*dims[0];
#endif
}

/// @brief Similar to Idx2Dim, but wraps around if ijk exceeds any bounds of dims. wasPeriodic is set to true if the index wrapper around, and false otherwise. \see Idx2Dim Dim2Idx
/// @param dims a 2-D or 3-D array of dimensions
/// @param ijk a 2-D or 3-D array of indices
/// @param wasPeriodic set to true if index wrapped around and false otherwise
/// @author WVN
static inline int Idx2DimPeriodic(int* dims, int* ijk, char* wasPeriodic)
{
    char output = 0;
    int delta[CMF_DIM];
    for (int d = 0; d < CMF_DIM; d++) delta[d] = 0;
    for (int d = 0; d < CMF_DIM; d++) delta[d] = (ijk[d]<0)?dims[d]:delta[d];
    for (int d = 0; d < CMF_DIM; d++) delta[d] = (ijk[d]>=dims[d])?-dims[d]:delta[d];
    for (int d = 0; d < CMF_DIM; d++) output += (delta[d]==0)?0:(1<<d);
    *wasPeriodic = output;
#if(CMF_IS3D)
    return (ijk[0]+delta[0]) + (ijk[1]+delta[1])*dims[0] + (ijk[2]+delta[2])*dims[1]*dims[0];
#else
    return (ijk[0]+delta[0]) + (ijk[1]+delta[1])*dims[0];
#endif
}

/// @brief Returns a char-vector based on the octant (in 3-D) or quadrant (in 2-D) of coords within a block defined by bounds, where coords = (x, y, [z]) and bounds = (xmin, xmax, ymin, ymax, [zmin], [zmax]).
/// For example, for bounds = (0, 1, 0, 1, 0, 1) and coords = (0.4, 0.9, 0.1), the return value is 3, or 010.
/// @param bounds An array specifying the block boundaries (xmin, xmax, ymin, ymax, [zmin], [zmax])
/// @param An array specifying the coordinates (x, y, [z])
/// @author WVN
static inline char GetOctant(double* bounds, double* coords)
{
#if(CMF_IS3D)
    return ((2.0*(coords[0]-bounds[0])>=(bounds[1]-bounds[0]))?1:0)
        +  ((2.0*(coords[1]-bounds[2])>=(bounds[3]-bounds[2]))?2:0)
        +  ((2.0*(coords[2]-bounds[4])>=(bounds[5]-bounds[4]))?4:0);
#else
    return ((2.0*(coords[0]-bounds[0])>=(bounds[1]-bounds[0]))?1:0)+((2.0*(coords[1]-bounds[2])>=(bounds[3]-bounds[2]))?2:0);
#endif
}

/// @brief Check if coords is contained within a box defined by bounds, where coords = (x, y, [z]) and bounds = (xmin, xmax, ymin, ymax, [zmin], [zmax]).
/// @param bounds An array specifying the block boundaries (xmin, xmax, ymin, ymax, [zmin], [zmax])
/// @param An array specifying the coordinates (x, y, [z])
/// @author WVN
static inline bool BoxContains(double* bounds, double* coords)
{
#if(CMF_IS3D)
    return (coords[0]>bounds[0])&&(coords[0]<bounds[1])&&(coords[1]>bounds[2])&&(coords[1]<bounds[3])&&(coords[2]>bounds[4])&&(coords[2]<bounds[5]);
#else
    return (coords[0]>bounds[0])&&(coords[0]<bounds[1])&&(coords[1]>bounds[2])&&(coords[1]<bounds[3]);
#endif
}

/// @brief Provides a random integer strictly less than bound.
/// @param bound The exclusive maximum return value
/// @author WVN
static inline int RandomInt(int bound)
{
    double r = (double)(rand()%0x000fffff)/((double)0x000fffff);
    return (int)(r*bound)%bound;
}

/// @brief Interprets an integer as a vector of three chars and a char as a vector of three bits, then returns the linear combination.
/// for example, basis = 0x00010203 and coord = 0b00000011, then the output is 0*3 + 1*2 + 1*1 = 2. \see BasisEvalReverse
/// @param basis interpreted as three chars aa, bb, cc, with basis = 0x00aabbcc
/// @param coord interpreted as three bits a, b, c, with coord = 0b00000abc
/// @author WVN
static inline char BasisEval(int basis, char coord)
{
    return (char)((coord&1)*((basis&0x000000ff)>>0) + ((coord&2)>>1)*((basis&0x0000ff00)>>8) + ((coord&4)>>2)*((basis&0x00ff0000)>>16));
}

/// @brief Interprets an integer as a vector of three chars and a char as a vector of three bits, then returns the linear combination.
/// for example, basis = 0x00010203 and coord = 0b00000011, then the output is 1*3 + 1*2 + 0*1 = 5. \see BasisEval
/// @param basis interpreted as three chars aa, bb, cc, with basis = 0x00aabbcc
/// @param coord interpreted as three bits a, b, c, with coord = 0b00000abc
/// @author WVN
static inline char BasisEvalReverse(int basis, char coord)
{
    return (char)((coord&1)*((basis&0x00ff0000)>>16) + ((coord&2)>>1)*((basis&0x0000ff00)>>8) + ((coord&4)>>2)*((basis&0x000000ff)>>0));
}

/// @brief Prints an int according to the interpretation found in BasisEval. Used only for debugging. \see BasisEval BasisEvalReverse
/// @param p interpreted as three chars aa, bb, cc, with p = 0x00aabbcc
/// @author WVN
static inline void PrintIntAsVector(int p)
{
    cmf::cmfout << (int)((p&0xff000000)>>24) << ", ";
    cmf::cmfout << (int)((p&0x00ff0000)>>16) << ", ";
    cmf::cmfout << (int)((p&0x0000ff00)>>8 ) << ", ";
    cmf::cmfout << (int)((p&0x000000ff)>>0 ) << cmf::cmfendl;
}

/// @brief Prints a char according to the interpretation found in BasisEval. Used only for debugging. \see BasisEval BasisEvalReverse
/// @param p interpreted as three bits a, b, c, with p = 0b00000abc
/// @author WVN
static inline void PrintCharAsVector(char p)
{
    std::cout << ((p&4)>>2) << ", ";
    std::cout << ((p&2)>>1) << ", ";
    std::cout << ((p&1)>>0) << std::endl;
}

#endif
