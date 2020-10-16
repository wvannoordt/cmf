#ifndef CMF_PNG_IMAGE_H
#define CMF_PNG_IMAGE_H

#include <string>
#include "PngCrc32.h"
#include "CmfError.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include "CmfZlib.h"
#include <vector>
#include <assert.h>
#include <cstring>

namespace cmf
{
    /// @brief A type representing the color of a single pixel. More convenient than an integer. BGRA format.
    typedef unsigned char pxtype[4];
    
    /// @brief Class that represents a PNG image that can be written to disk.
	/// @author WVN
    class PngImage
    {
        public:
            /// @brief Constructor for PngImage, where the pixel buffer is provided externally.
            /// \pre imageBuffer_in should have size height_in by width_in.
            /// \pre imageBuffer_in should contain data in BGRA format (4x char)
            /// @param height_in The height of the pixel buffer
            /// @param width_in The width of the pixel buffer
            /// @param imageBuffer_in The underlying (externally managed) pixel buffer
        	/// @author WVN
            PngImage(int height_in, int width_in, int* imageBuffer_in);
            
            /// @brief Constructor for PngImage, where the pixel buffer is provided internally.
            /// @param height_in The height of the pixel buffer
            /// @param width_in The width of the pixel buffer
        	/// @author WVN
            PngImage(int height_in, int width_in);
            
            /// @brief Fills the image with the specified color.
            /// @param color The color to fill the image with
        	/// @author WVN
            void Fill(int color);
            
            /// @brief Fills the image with the specified color.
            /// @param color The color to fill the image with
        	/// @author WVN
            void Fill(pxtype color);
            
            /// @brief Sets a pixel in the pixel buffer
            /// @param row Row-index of target pixel
            /// @param col Column-index of target pixel
            /// @param color The color to set the pixel to
        	/// @author WVN
            void SetPixel(int row, int col, int color);
            
            /// @brief Sets a pixel in the pixel buffer
            /// @param row Row-index of target pixel
            /// @param col Column-index of target pixel
            /// @param color The color to set the pixel to
        	/// @author WVN
            void SetPixel(int row, int col, pxtype color);
            
            /// @brief Destructor for PngImage. Frees resources if needed.
        	/// @author WVN
            ~PngImage(void);
            
            /// @brief Writes the PNG image to a file
            /// @param filename The name of the file to be written
            /// \pre filename should have extension ".png"
        	/// @author WVN
            void Write(std::string filename);
            
            /// @brief Returns the underlying pixel buffer.
        	/// @author WVN
            int* GetBuffer(void);
        
        private:
            /// @brief Writes the PNG signature to the file pointer.
            /// @param fileWriter File Pointer to be written to
        	/// @author WVN
            void WriteSignature(FILE* fileWriter);
            
            /// @brief Writes the PNG header to the file pointer.
            /// @param fileWriter File Pointer to be written to
            /// @param width Image width
            /// @param height Image height
        	/// @author WVN
            void WriteHeader(FILE* fileWriter, const int& width, const int& height);
            
            /// @brief Writes the PNG data to the file pointer.
            /// @param fileWriter File Pointer to be written to
            /// @param col Image width
            /// @param row Image height
            /// @param imdata underlying image buffer
        	/// @author WVN
            void WriteData(FILE* fileWriter, const int& col, const int& row, int* imdata);
            
            /// @brief Writes the PNG end material to the file pointer.
            /// @param fileWriter File Pointer to be written to
        	/// @author WVN
            void WriteEnd(FILE* fileWriter);
            
            /// @brief Reverses the bytes of an integer.
            /// @param i Pointer to integer to be reversed
        	/// @author WVN
            void FlipEndianness(int* i);
            
            /// @brief Compresses data using ZLIB if enabled
            /// @param in_data Buffer of data to be compressed
            /// @param in_data_size Size of data to be compressed
            /// @param out_data Compressed output
        	/// @author WVN
            void CompressMemory(void *in_data, size_t in_data_size, std::vector<uint8_t> &out_data);
            
            /// @brief The width of the image buffer
            int width;
            
            /// @brief The height of the image buffer
            int height;
            
            /// @brief The underlying pixel buffer in (R, G, B, A) format
            int* imageBuffer;
            
            /// @brief Indicates whether the underling buffer is provided externally
            bool hasSelfContainedBuffer;
            
            /// @brief Indicates whether the underling buffer is freed on object destruction
            bool bufferRequiresDealloc;
            
            /// @brief Indicates whether the current machine is <a href="https://en.wikipedia.org/wiki/Endianness">big-endian</a>
            bool machineIsBigEndian;
            
            /// @brief Object used to compute CRC-32 checksums.
            PngCrc32 checksum;
    };
}

#endif