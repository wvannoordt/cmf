#ifndef CMF_INTERPOLATION_OPERATOR_1D_H
#define CMF_INTERPOLATION_OPERATOR_1D_H

#include <vector>
#include "CmfGC.h"

namespace cmf
{
    ///@brief Represents a 1-D interpolation operator using lagrange interpolation
    ///@author WVN
    struct InterpolationOperator1D
    {
        ///@brief The supporting coordinates
        double* coords = NULL;
        
        ///@brief The values providing the interpolation
        double* data = NULL;
        
        ///@brief The number of coefficients
        unsigned int size = 0;
        
        ///@brief The order of interpolation
        int order = 0;
        
        ///@brief Sets the size property and reallocates the coordinate and data buffer. This does not copy any existing data from old buffers.
        ///@param size_in the number of elements to resize to
        ///@author WVN
        void SetSize(unsigned int size_in)
        {
            size = size_in;
            Free();
            coords = (double*)Cmf_Alloc(size*sizeof(double));
            data = (double*)Cmf_Alloc(size*sizeof(double));
        }
        
        ///@brief Fills the data buffer with the given value
        ///@param val the value to set
        ///@author WVN
        void FillData(double val)
        {
            for (int i = 0; i < size; i++)
            {
                data[i] = val;
            }
        }
        
        ///@brief Given a coordinate value and an index in the coordinate list,
        ///returns the interpolation coefficient associated with the data value at that coordinate with respect to the provided value
        ///@param stencilMinIndex The index (relative to coords) that is the minimal point on the interpolation stencil
        ///@param index the index
        ///@param coord The coordinate value
        ///@author WVN
        inline double GetCoefficientAtPoint(int stencilMinIndex, int offset, double coord)
        {
            
            double output = 1.0;
            for (int i = 0;        i < offset; i++)
            {
                output *= (coord-coords[stencilMinIndex+i])/(coords[stencilMinIndex+offset]-coords[stencilMinIndex+i]);
            }
            for (int i = offset+1; i < order;  i++)
            {
                output *= (coord-coords[stencilMinIndex+i])/(coords[stencilMinIndex+offset]-coords[stencilMinIndex+i]);
            }
            return output;
        }
        
        ///@brief Given a coordinate value, searches for the lowest index of the interpolation stencil
        ///@param coord The coordinate value
        ///@author WVN
        int FindMinStencilIndex(double coord)
        {
            //This is a slow implementation, but the maximum expected value for size is like 7
            int j = 0;
            for (int i = 0; i < size; i++)
            {
                j=i;
                if (coords[i] > coord)
                {
                    break;
                }
            }
            if (j>size-(order+1)/2)
            {
                j = size-(order+1)/2;
            }
            return ((j-order/2)<0)?0:(j-order/2);
        }
        
        ///@brief Frees the data buffers if they are not NULL: sets them to null
        ///@author WVN
        void Free(void)
        {
            if (coords != NULL) Cmf_Free(coords);
            if (data != NULL) Cmf_Free(data);
            data = NULL;
            coords = NULL;
        }
        
        ///@brief Destructor (frees buffers)
        ///@author WVN
        ~InterpolationOperator1D(void)
        {
            Free();
        }
    };
}

#endif