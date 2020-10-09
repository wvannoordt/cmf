#ifndef VTK_BUFFER_H
#define VTK_BUFFER_H
#include "VtkFile.h"
namespace cmf
{
    class VtkBuffer
    {
        /// @brief Class representing a general VTK data buffer that can be written to and read from
        /// @author WVN
        public:
            
            /// @brief Constructor for VtkBuffer
            /// @param input a VtkAttributable object that a buffer is being provided to
            /// \see VtkAttributable
            /// @author WVN
            VtkBuffer(VtkAttributable* input);
            
            /// @brief Destructor for VtkBuffer, frees resources automatically
            /// @author WVN
            ~VtkBuffer(void);
            
            /// @brief Writes a double to the underlying buffer
            /// @param x Value to be written to the underlying buffer
            /// @author WVN
            void operator << (double x);
            
            /// @brief Writes an integer to the underlying buffer
            /// @param x Value to be written to the underlying buffer
            /// @author WVN
            void operator << (int x);
            
            /// @brief Deprecated
            /// @author WVN
            int Next(void);
        private:
            /// @brief The underlying VtkAttributable object that the buffer pertains to
            VtkAttributable* component;
            
            /// @brief The buffer for storing the data
            char* allocatedBuffer;
            
            /// @brief The allocated size of allocatedBuffer
            size_t bufferSize;
            
            /// @brief The current index offset (with respect to allocatedBuffer)
            size_t curentOffset;
            
            /// @brief Deprecated
            int idx;
    };
}

#endif
