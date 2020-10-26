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

/// @brief Transforms 1-D index into 3-D index.
/// @param idx 1-D index
/// @param dims 3-D size
/// @param ijk  (output) resulting 3-D index
/// @author WVN
static inline void Dim2Idx3(int idx, int* dims, int* ijk)
{
    ijk[0] = idx%(dims[0]);
    ijk[1] = ((idx-ijk[0]) / dims[0])%dims[1];
    ijk[2] = ((idx-ijk[0]-dims[0]*ijk[1])/(dims[1]*dims[0])) % dims[2];
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

/// @brief Returns a 1-D index based on a 3-D array. \see Idx2Dim
/// @param dims a 3-D array of dimensions
/// @param ijk a 3-D array of indices
/// @author WVN
static inline int Idx2Dim3(int* dims, int* ijk)
{
    return ijk[0] + ijk[1]*dims[0] + ijk[2]*dims[1]*dims[0];
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
/// @param coords An array specifying the coordinates (x, y, [z])
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


/// @brief Check if indices are within specified bounds
/// @param ijk An array of 3 indices
/// @param bounds An array of 3 index bounds
/// @author WVN
static inline bool CheckBounds3(int* ijk, int* bounds)
{
    return ((ijk[0]>=0) && (ijk[0]<bounds[0]) && (ijk[1]>=0) && (ijk[1]<bounds[1]) && (ijk[2]>=0) && (ijk[2]<bounds[2]));
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

/// @brief Check if coords is contained within a box defined by bounds, where coords = (x, y, z) and bounds = (xmin, xmax, ymin, ymax, zmin, zmax).
/// @param bounds An array specifying the block boundaries (xmin, xmax, ymin, ymax, zmin, zmax)
/// @param An array specifying the coordinates (x, y, z)
/// @author WVN
static inline bool BoxContains3(double* bounds, double* coords)
{
    return (coords[0]>bounds[0])&&(coords[0]<bounds[1])&&(coords[1]>bounds[2])&&(coords[1]<bounds[3])&&(coords[2]>bounds[4])&&(coords[2]<bounds[5]);
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

/// @brief Checks if a plane intersects a box
/// @param point
/// @param normalVec Normal vector
/// @param box bounds of the box (xmin, xmax, ymin, ymax, zmin, zmax)
/// \pre All arrays must have size 3
/// @author WVN
static inline bool PlaneIntersectsBox(double* point, double* normalVec, double* box)
{
    //strictly speaking, only need to test the "most akigned" vertices
    bool hasPositive = false;
	bool hasNegative = false;
    double corner[3];
    double dotProd;
	for (char b = 0; b < 8; b++)
	{
		corner[0] = box[0 + ((b>>0)&1)];
        corner[1] = box[2 + ((b>>1)&1)];
        corner[2] = box[4 + ((b>>2)&1)];
        dotProd  = normalVec[0]*(corner[0] - point[0]);
        dotProd += normalVec[1]*(corner[1] - point[1]);
        dotProd += normalVec[2]*(corner[2] - point[2]);
        hasPositive = hasPositive||(dotProd>0);
        hasNegative = hasNegative||(dotProd<0);
        if ((-1e-8 < dotProd) && (dotProd < 1e-8)) return true;
        if (hasNegative&&hasPositive) return true;
	}
    return false;
}

/// @brief Checks if two boxes intersect
/// @param b1 First box (xmin, xmax, ymin, ymax, zmin, zmax)
/// @param b2 First box (xmin, xmax, ymin, ymax, zmin, zmax
/// \pre All arrays must have size 3
/// @author WVN
static inline bool BoxOverlap3(double* b1, double* b2)
{
    return ((b1[0] <= b2[1]) || (b1[1] >= b2[0])) && ((b1[2] <= b2[3]) || (b1[3] >= b2[2])) && ((b1[4] <= b2[5]) || (b1[5] >= b2[4]));
}

// see http://fileadmin.cs.lth.se/cs/personal/tomas_akenine-moller/code/tribox_tam.pdf for algorithm below

/// @brief Checks if a triangle intersects a box
/// @param p1 First point
/// @param p2 First point
/// @param p3 First point
/// @param normalVec Normal vector
/// @param box bounds of the box (xmin, xmax, ymin, ymax, zmin, zmax)
/// \pre All arrays must have size 3
/// @author WVN
static inline bool TriangleIntersectsBox(double* p1, double* p2, double* p3, double* nvec, double* box)
{
    double tbox[6];
    double c[3];
    c[0] = 0.5*(box[1]+box[0]);
    c[1] = 0.5*(box[3]+box[2]);
    c[2] = 0.5*(box[5]+box[4]);
    double h[3];
    h[0] = 0.5*(box[1]-box[0]);
    h[1] = 0.5*(box[3]-box[2]);
    h[2] = 0.5*(box[5]-box[4]);
    tbox[0] = CMFMIN(CMFMIN(p1[0], p2[0]), p3[0]);
    tbox[1] = CMFMAX(CMFMAX(p1[0], p2[0]), p3[0]);
    tbox[2] = CMFMIN(CMFMIN(p1[1], p2[1]), p3[1]);
    tbox[3] = CMFMAX(CMFMAX(p1[1], p2[1]), p3[1]);
    tbox[4] = CMFMIN(CMFMIN(p1[2], p2[2]), p3[2]);
    tbox[5] = CMFMAX(CMFMAX(p1[2], p2[2]), p3[2]);
    if (!BoxOverlap3(tbox, box)) return false;
    if (!PlaneIntersectsBox(p1, nvec, box)) return false;
    double ei[3];
    double fj[3];
    double aij[3];
    double p[3];
    double r;
    int i, j;
    for (int z = 0; z < 9; z++)
    {
        i = z/3;
        j = z%3;
        ei[0] = 0.0; ei[1] = 0.0; ei[2] = 0.0;
        fj[0] = (j==0)?(p2[0]-p1[0]):((j==1)?(p3[0]-p2[0]):(p1[0]-p3[0]));
        fj[1] = (j==0)?(p2[1]-p1[1]):((j==1)?(p3[1]-p2[1]):(p1[1]-p3[1]));
        fj[2] = (j==0)?(p2[2]-p1[2]):((j==1)?(p3[2]-p2[2]):(p1[2]-p3[2]));
        ei[i] = 1.0;
        aij[0] = ei[1]*fj[2] - ei[2]*fj[1];
        aij[1] = ei[0]*fj[2] - ei[2]*fj[0];
        aij[2] = ei[0]*fj[1] - ei[1]*fj[0];
        p[0] = aij[0]*(p1[0]-c[0]) + aij[1]*(p1[1]-c[1]) + aij[2]*(p1[2]-c[2]);
        p[1] = aij[0]*(p2[0]-c[0]) + aij[1]*(p2[1]-c[1]) + aij[2]*(p2[2]-c[2]);
        p[2] = aij[0]*(p3[0]-c[0]) + aij[1]*(p3[1]-c[1]) + aij[2]*(p3[2]-c[2]);
        r = h[0]*__d_abs(aij[0]) + h[1]*__d_abs(aij[1]) + h[2]*__d_abs(aij[2]);
        if ((r < CMFMIN(CMFMIN(p[0], p[1]), p[2])) || (CMFMAX(CMFMAX(p[0], p[1]), p[2]) < -r)) return false;
    }
    return true;
}

#endif
