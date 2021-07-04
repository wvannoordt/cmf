#ifndef CMF_VEC3_H
#define CMF_VEC3_H
#include <type_traits>
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
        
        /// @brief Constructor
        /// @param x value to initialize all elements with
        /// @author WVN
        template <typename othertype> Vec3(othertype x) {v[0] = (othertype)x; v[1] = (othertype)x; v[2] = (othertype)x;}
        
        /// @brief Constructor, using initial and final values
        /// @param ini Pointer (of size > 3) to the coordinates of initial point
        /// @param ter Pointer (of size > 3) to the coordinates of terminal point
        /// @author WVN
        Vec3(numericType* ini, numericType* ter) {v[0] = ter[0]-ini[0]; v[1] = ter[1]-ini[1]; v[2] = ter[2]-ini[2];}
        
        /// @brief Copy constructor
        /// @param w Vector to copy
        /// @author WVN
        Vec3(const Vec3& w) {v[0] = w.v[0]; v[1] = w.v[1]; v[2] = w.v[2];}
        
        /// @brief index operator
        /// @param i index
        /// @author WVN
        numericType & operator [] (int i) {return *(v+i);}
        
        ///@brief the data
        numericType v[3];
        
        ///@brief Normalize in-place by L2-norm
        void Normalize(void)
        {
            numericType norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            v[0]/=norm;
            v[1]/=norm;
            v[2]/=norm;
        }
        
        /// @brief Returns a string representing the vector
        std::string str()
        {
            std::string output = "[";
            output += std::to_string(v[0]) + ", ";
            output += std::to_string(v[1]) + ", ";
            output += std::to_string(v[2]) + "]";
            return output;
        }
        
        ///@brief Addition operator
        Vec3 operator + (const Vec3& w) {return Vec3(v[0]+w.v[0], v[1]+w.v[1], v[2]+w.v[2]);}
        
        ///@brief Difference operator
        Vec3 operator - (const Vec3& w) {return Vec3(v[0]-w.v[0], v[1]-w.v[1], v[2]-w.v[2]);}
        
        ///@brief Vector cross-product, a%b returns a "cross" b
        Vec3 operator % (const Vec3& w) {return Vec3(v[1]*w.v[2] - v[2]*w.v[1], v[2]*w.v[0] - v[0]*w.v[2], v[0]*w.v[1] - v[1]*w.v[0]);}
        
        ///@brief Scalar multiplication
        Vec3 operator * (const numericType a) {return Vec3(a*v[0], a*v[1], a*v[2]);}
        
        ///@brief Scalar multiplication
        Vec3& operator *= (const numericType a) {v[0]*=a; v[1]*=a; v[2]*=a; return *this;}
        
        ///@brief Increment operator
        Vec3& operator += (Vec3& rhs) {v[0]+=rhs[0]; v[1]+=rhs[1]; v[2]+=rhs[2]; return *this;}
        
        ///@brief Decrement operator
        Vec3& operator -= (Vec3& rhs) {v[0]-=rhs[0]; v[1]-=rhs[1]; v[2]-=rhs[2]; return *this;}
        
        ///@brief Scalar division
        Vec3 operator / (const numericType a) {return Vec3(v[0]/a, v[1]/a, v[2]/a);}
        
        ///@brief L2 inner product
        numericType operator * (const Vec3& w) {return v[0]*w.v[0] + v[1]*w.v[1] + v[2]*w.v[2];}
        
        ///@brief L2 norm
        numericType Norm(void) {return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);}
        
        Vec3& operator = (const numericType rhs)
        {
            v[0] = rhs;
            v[1] = rhs;
            v[2] = rhs;
            return *this;
        }
    };
    
    /// @brief A simple N-D vector class
    /// \pre Note: use this class sparingly, as it is not optimized
    /// @author WVN
    template <typename numericType, const int VECDIM> struct Vec
    {
        /// @brief Zero constructor
        /// @author WVN
        Vec(void) {for (int i = 0; i < VECDIM; i++) v[i] = 0;}
        
        /// @brief Zero constructor
        /// @author WVN
        Vec(std::vector<numericType> inpt) {for (int i = 0; i < VECDIM; i++) v[i] = inpt[i];}
        
        /// @brief Constructor
        /// @param x Pointer (of size > VECDIM)
        /// @author WVN
        Vec(numericType* x) {for (int i = 0; i < VECDIM; i++) v[i] = x[i];}
        
        /// @brief Constructor
        /// @param x Pointer (of size > VECDIM)
        /// @author WVN
        template <typename othertype> Vec(othertype x) {for (int i = 0; i < VECDIM; i++) v[i] = (numericType)x;}
        
        
        /// @brief Variadic Constructor helper function
        /// @param lev the recursive level
        /// @param t The next element
        /// @param ts The elements of the vector
        /// @author WVN
        template <typename T> void vrset(int lev, T t)
    	{
    		v[lev] = (numericType)t;
    	}
        
        /// @brief Variadic Constructor helper function
        /// @param lev the recursive level
        /// @param t The next element
        /// @param ts The elements of the vector
        /// @author WVN
        template <typename T, typename... Ts> void vrset(int lev, T t, Ts... ts)
    	{
    		v[lev] = (numericType)t;
    		vrset(lev+1, ts...);
    	}
        
        /// @brief Variadic Constructor
        /// @param ts The elements of the vector
        /// @author WVN
        template <typename... Ts> Vec(Ts... ts)
    	{
            static_assert(sizeof...(Ts)==VECDIM, "Incorrect rank of vector assignment");
    		vrset(0, ts...);
    	}
        
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
        numericType v[VECDIM];
        
        /// @brief assignment operator
        /// @param rhs the data to assign
        /// @author WVN
        template <typename othertype> Vec& operator = (const othertype rhs)
        {
            for (int i = 0; i < VECDIM; i++) v[i] = (numericType)rhs;
            return *this;
        }
        
        /// @brief index operator
        /// @param i index
        /// @author WVN
        numericType & operator [] (int i) {return *(v+i);}
        
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