#ifndef CMF_PNG_CRC32_H
#define CMF_PNG_CRC32_H
#include <iostream>
#define CRC32_TABLE_CT 256
namespace cmf
{
    typedef unsigned char ubyte;
    /// @brief Class for computing CRC-32 checksums
	/// @author WVN
    class PngCrc32
    {
        public:
            /// @brief Constuctor for PngCrc32
        	/// @author WVN
            PngCrc32(void);
            
            /// @brief Destructor for PngCrc32
        	/// @author WVN
            ~PngCrc32(void);
            
            /// @brief Computes the <a href="https://en.wikipedia.org/wiki/Cyclic_redundancy_check">CRC-32 checksum</a> for a given buffer
            /// @param bytes The buffer to be checksummed
            /// @param length The size of bytes
        	/// @author WVN
            unsigned int ComputeChecksum(ubyte* bytes, size_t length);
        private:
            
            /// @brief A table to store polynomial coefficients
            unsigned int table[CRC32_TABLE_CT];
    };
}

#endif