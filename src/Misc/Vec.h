#ifndef TRX_VEC3_H
#define TRX_VEC3_H
#include <iostream>
#include <cmath>
namespace cmf
{
    /// @brief A simple 3D vector class
    /// \pre Note: use this class sparingly, as it is not optimized
    /// @author WVN
    template <typename numericType> struct Vec3
    {
        /// @brief Zero constructor
        /// @author WVN
        Vec3(void) {v[0] = 0; v[1] = 0; v[2] = 0;}
        
        /// @brief Constructor
        /// @param x X component
        /// @param y Y component
        /// @param z Z component
        /// @author WVN
        Vec3(numericType x, numericType y, numericType z) {v[0] = x; v[1] = y; v[2] = z;}
        
        /// @brief Constructor
        /// @param x Pointer (of size > 3)
        /// @author WVN
        Vec3(numericType* x) {v[0] = x[0]; v[1] = x[1]; v[2] = x[2];}
        
        /// @brief Constructor, using initial and final values
        /// @param ini Pointer (of size > 3) to the coordinates of initial point
        /// @param ter Pointer (of size > 3) to the coordinates of terminal point
        /// @author WVN
        Vec3(numericType* ini, numericType* ter) {v[0] = ter[0]-ini[0]; v[1] = ter[1]-ini[1]; v[2] = ter[2]-ini[2];}
        
        /// @brief Copy constructor
        /// @param w Vector to copy
        /// @author WVN
        Vec3(const Vec3& w) {v[0] = w.v[0]; v[1] = w.v[1]; v[2] = w.v[2];}
        
        ///@brief the data
        double v[3];
        
        ///@brief Normalize in-place by L2-norm
        void Normalize(void)
        {
            numericType norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            v[0]/=norm;
            v[1]/=norm;
            v[2]/=norm;
        }
        
        ///@brief Addition operator
        Vec3 operator + (const Vec3& w) {return Vec3(v[0]+w.v[0], v[1]+w.v[1], v[2]+w.v[2]);}
        
        ///@brief Difference operator
        Vec3 operator - (const Vec3& w) {return Vec3(v[0]-w.v[0], v[1]-w.v[1], v[2]-w.v[2]);}
        
        ///@brief Vector cross-product, a%b returns a "cross" b
        Vec3 operator % (const Vec3& w) {return Vec3(v[1]*w.v[2] - v[2]*w.v[1], v[2]*w.v[0] - v[0]*w.v[2], v[0]*w.v[1] - v[1]*w.v[0]);}
        
        ///@brief Scalar multiplication
        Vec3 operator * (const numericType a) {return Vec3(a*v[0], a*v[1], a*v[2]);}
        
        ///@brief Scalar division
        Vec3 operator / (const numericType a) {return Vec3(v[0]/a, v[1]/a, v[2]/a);}
        
        ///@brief L2 inner product
        numericType operator * (const Vec3& w) {return v[0]*w.v[0] + v[1]*w.v[1] + v[2]*w.v[2];}
        
        ///@brief L2 norm
        numericType Norm(void) {return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);}
    };
    
    /// @brief A simple N-D vector class
    /// \pre Note: use this class sparingly, as it is not optimized
    /// @author WVN
    template <typename numericType, const int VECDIM> struct Vec
    {
        /// @brief Zero constructor
        /// @author WVN
        Vec(void) {for (int i = 0; i < VECDIM; i++) v[i] = 0;}
        
        /// @brief Constructor
        /// @param x Pointer (of size > VECDIM)
        /// @author WVN
        Vec(numericType* x) {for (int i = 0; i < VECDIM; i++) v[i] = x[i];}
        
        /// @brief Constructor, using initial and final values
        /// @param ini Pointer (of size > VECDIM) to the coordinates of initial point
        /// @param ter Pointer (of size > VECDIM) to the coordinates of terminal point
        /// @author WVN
        Vec(numericType* ini, numericType* ter) {for (int i = 0; i < VECDIM; i++) v[i] = ter[i]-ini[i];}
        
        /// @brief Copy constructor
        /// @param w Vector to copy
        /// @author WVN
        Vec(const Vec& w) {for (int i = 0; i < VECDIM; i++) v[i] = w.v[i];}
        
        ///@brief the data
        double v[VECDIM];
        
        ///@brief Normalize in-place by L2-norm
        void Normalize(void)
        {
            numericType norm = 0;
            for (int i = 0; i < VECDIM; i++) norm += v[i]*v[i];
            norm = sqrt(norm);
            for (int i = 0; i < VECDIM; i++) v[i]/=norm;
        }
    };
    
    ///@brief Allows streaming of Vec3<float>
    static std::ostream & operator<<(std::ostream & os, const Vec3<float> & vec)
    {
       os << "[" << vec.v[0] << ", " << vec.v[1] << ", " << vec.v[2] << "]";
       return os;
    }
    
    ///@brief Allows streaming of Vec3<float>
    static std::ostream & operator<<(std::ostream & os, const Vec3<int> & vec)
    {
       os << "[" << vec.v[0] << ", " << vec.v[1] << ", " << vec.v[2] << "]";
       return os;
    }
    
    ///@brief Allows streaming of Vec3<float>
    static std::ostream & operator<<(std::ostream & os, const Vec3<double> & vec)
    {
       os << "[" << vec.v[0] << ", " << vec.v[1] << ", " << vec.v[2] << "]";
       return os;
    }
    
    ///@brief Allows streaming of Vec3<float>
    template <typename numericType, const int VECDIM> static std::ostream & operator<<(std::ostream & os, const Vec<numericType, VECDIM> & vec)
    {
        if (VECDIM<1)
        {
            os << "[]";
            return os;
        }
       os << "[" << vec.v[0];
       for (int i = 1; i < VECDIM; i++) os << ", " << vec.v[i];
       os << "]";
       return os;
    }
}

#endif