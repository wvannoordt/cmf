#ifndef CMF_PRINT_DEBUG_H
#define CMF_PRINT_DEBUG_H
#include <iostream>
namespace cmf
{
    
    /// @brief Printing function (base case). FOR DEBUGGING ONLY!!
    /// @author WVN
    /// @param t an item to print
    template <typename T> void print_recursive(T t)
    {
        std::cout << t << std::endl;
    }
    
    /// @brief Printing function (recursive case). FOR DEBUGGING ONLY!!
    /// @author WVN
    /// @param t an item to print
    /// @param ts The list of things to print
    template <typename T, typename... Ts> void print_recursive(T t, Ts... ts)
    {
        std::cout << t << " ";
        print_recursive(ts...);
    }
    
    /// @brief Printing function. FOR DEBUGGING ONLY!!
    /// @author WVN
    /// @param ts The list of things to print
    template <typename... Ts> void print(Ts... ts)
    {
        print_recursive(ts...);
    }
}

#endif