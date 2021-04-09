#ifndef CMF_PRINT_DEBUG_H
#define CMF_PRINT_DEBUG_H
#include <iostream>
#include <ostream>
namespace cmf
{
    
    /// @brief Printing function (base case). FOR DEBUGGING ONLY!!
    /// @author WVN
    /// @param t an item to print
    template <typename T> void print_recursive(std::ostream& stm, T t)
    {
        stm << t << std::endl;
    }
    
    /// @brief Printing function (recursive case). FOR DEBUGGING ONLY!!
    /// @author WVN
    /// @param t an item to print
    /// @param ts The list of things to print
    template <typename T, typename... Ts> void print_recursive(std::ostream& stm, T t, Ts... ts)
    {
        stm << t << " ";
        print_recursive(stm, ts...);
    }
    
    /// @brief Printing function. FOR DEBUGGING ONLY!!
    /// @author WVN
    /// @param ts The list of things to print
    template <typename... Ts> void print(Ts... ts)
    {
        print_recursive(std::cout, ts...);
    }
    
    /// @brief Prints arbitrary data to a stream
    /// @author WVN
    /// @param ts The list of things to print
    template <typename... Ts> void PrintToStream(std::ostream& stm, Ts... ts)
    {
        print_recursive(stm, ts...);
    }
}

#endif