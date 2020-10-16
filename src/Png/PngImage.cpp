#include "PngImage.h"
#define APH_SHIFT 24
#define RED_SHIFT 16
#define GRN_SHIFT 8
#define BLU_SHIFT 0
#define rgb2int(red, grn, blu) ((255 << APH_SHIFT) | ((int)(255*(red)) << RED_SHIFT) | ((int)(255*(grn)) << GRN_SHIFT) | ((int)(255*(blu)) << BLU_SHIFT))
#define int2red(myval) (ubyte)( ((myval)&(0x000000ff << RED_SHIFT)) >> RED_SHIFT)
#define int2grn(myval) (ubyte)( ((myval)&(0x000000ff << GRN_SHIFT)) >> GRN_SHIFT)
#define int2blu(myval) (ubyte)( ((myval)&(0x000000ff << BLU_SHIFT)) >> BLU_SHIFT)
namespace cmf
{
    PngImage::PngImage(int height_in, int width_in, int* imageBuffer_in)
    {
        height = height_in;
        width  = width_in;
        imageBuffer = imageBuffer_in;
        hasSelfContainedBuffer = false;
        bufferRequiresDealloc = false;
        int num = 1;
        machineIsBigEndian = ! ( *(char *)&num == 1 );
    }

    PngImage::PngImage(int height_in, int width_in)
    {
        height = height_in;
        width  = width_in;
        hasSelfContainedBuffer = true;
        bufferRequiresDealloc = true;
        imageBuffer = (int*)malloc(height * width * sizeof(int));
        int num = 1;
        machineIsBigEndian = ! ( *(char *)&num == 1 );
    }

    void PngImage::Write(std::string filename)
    {
#if(!CMF_ZLIB_EXT_ENABLE)
        CmfError("Tried to write PNG image without ZLIB extension enabled. Please recompile with ZLIB.");
#endif
        FILE* fileWriter;
        fileWriter = fopen(filename.c_str(), "w+b");
        WriteSignature(fileWriter);
        WriteHeader(fileWriter, width, height);
        WriteData(fileWriter, width, height, imageBuffer);
        WriteEnd(fileWriter);
        fclose(fileWriter);
    }

    int* PngImage::GetBuffer(void)
    {
        return imageBuffer;
    }

    PngImage::~PngImage(void)
    {
        if (bufferRequiresDealloc)
        {
            bufferRequiresDealloc = false;
            free(imageBuffer);
        }
    }

    void PngImage::Fill(int color)
    {
        for (size_t i = 0; i < width*height; i++) *(imageBuffer+i) = color;
    }

    void PngImage::Fill(pxtype color)
    {
        Fill(*((int*)color));
    }

    void PngImage::SetPixel(int row, int col, int color)
    {
        *(imageBuffer + row*width + col) = color;
    }

    void PngImage::SetPixel(int row, int col, pxtype color)
    {
        *((ubyte*)(imageBuffer + row*width + col)+0) = color[0];
        *((ubyte*)(imageBuffer + row*width + col)+1) = color[1];
        *((ubyte*)(imageBuffer + row*width + col)+2) = color[2];
        *((ubyte*)(imageBuffer + row*width + col)+3) = color[3];
    }

    void PngImage::WriteData(FILE* fileWriter, const int& col, const int& row, int* imdata)
    {

        ubyte* buffer_in;
        size_t input_buffer_size = ((3*col + 1) * row) * sizeof(ubyte);

        buffer_in = (ubyte*)malloc(input_buffer_size);
        //set line filter bytes

        int i, j;
        for (i = 0; i < row; i++) *(buffer_in + i*(3*col + 1)) = 0x00;

        //filter r g b r g b r g b ...
        for (i = 0; i < row; i++)
        {
            for (j = 0; j < col; j++)
            {
                *(buffer_in + i*(3*col + 1) + 3*j + 1) = int2red(*(imdata + i*col + j));
                *(buffer_in + i*(3*col + 1) + 3*j + 2) = int2grn(*(imdata + i*col + j));
                *(buffer_in + i*(3*col + 1) + 3*j + 3) = int2blu(*(imdata + i*col + j));
            }
        }

        std::vector<uint8_t> compressed_buffer_vec;
        CompressMemory(buffer_in, input_buffer_size, compressed_buffer_vec);
        int comp_size = static_cast<int>(compressed_buffer_vec.size());
        int comp_size_write = comp_size;
        if (!machineIsBigEndian) FlipEndianness(&comp_size_write);

        ubyte* comp_buffer_in = (ubyte*)malloc((4+comp_size)*sizeof(ubyte));
        std::copy(compressed_buffer_vec.begin(),compressed_buffer_vec.end(), comp_buffer_in + 4);
        *(comp_buffer_in+0) = 'I';
        *(comp_buffer_in+1) = 'D';
        *(comp_buffer_in+2) = 'A';
        *(comp_buffer_in+3) = 'T';

        fwrite(&comp_size_write, 1, sizeof(int), fileWriter);
        fwrite(comp_buffer_in, 1, (4+comp_size)*sizeof(ubyte), fileWriter);

        int crc32_data = checksum.ComputeChecksum(comp_buffer_in, 4+comp_size);
        if (!machineIsBigEndian) FlipEndianness(&crc32_data);
        fwrite(&crc32_data, 1, sizeof(int), fileWriter);
        free(comp_buffer_in);
        free(buffer_in);
    }

    void PngImage::CompressMemory(void *in_data, size_t in_data_size, std::vector<uint8_t> &out_data)
    {
#if(CMF_ZLIB_EXT_ENABLE)
        std::vector<uint8_t> buffer;

        const size_t BUFSIZE = 128 * 1024;
        uint8_t temp_buffer[BUFSIZE];

        z_stream strm;
        strm.zalloc = 0;
        strm.zfree = 0;
        strm.next_in = reinterpret_cast<uint8_t *>(in_data);
        strm.avail_in = in_data_size;
        strm.next_out = temp_buffer;
        strm.avail_out = BUFSIZE;

        deflateInit(&strm, Z_BEST_COMPRESSION);

        while (strm.avail_in != 0)
        {
            int res = deflate(&strm, Z_NO_FLUSH);
            assert(res == Z_OK);
            if (strm.avail_out == 0)
            {
                buffer.insert(buffer.end(), temp_buffer, temp_buffer + BUFSIZE);
                strm.next_out = temp_buffer;
                strm.avail_out = BUFSIZE;
            }
        }

        int deflate_res = Z_OK;
        while (deflate_res == Z_OK)
        {
            if (strm.avail_out == 0)
            {
                buffer.insert(buffer.end(), temp_buffer, temp_buffer + BUFSIZE);
                strm.next_out = temp_buffer;
                strm.avail_out = BUFSIZE;
            }
            deflate_res = deflate(&strm, Z_FINISH);
        }

        assert(deflate_res == Z_STREAM_END);
        buffer.insert(buffer.end(), temp_buffer, temp_buffer + BUFSIZE - strm.avail_out);
        deflateEnd(&strm);
        out_data.swap(buffer);
#else
        CmfError("Tried to write PNG image without ZLIB extension enabled. Please recompile with ZLIB.");
#endif
    }

    void PngImage::WriteSignature(FILE* fileWriter)
    {
        ubyte png_signature[8] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
        fwrite(&png_signature , sizeof(ubyte), 8*sizeof(ubyte), fileWriter);
    }
    void PngImage::WriteHeader(FILE* fileWriter, const int& width, const int& height)
    {
        int header_length = 13;
        int header_length_write = 13;

        ubyte header[header_length+4];

        int output_width = width;
        int output_height = height;
        ubyte bit_depth = 0x8;
        ubyte color_type = 0x2;
        ubyte compression_method = 0x0;
        ubyte interlace_method = 0x0;
        ubyte filter_method = 0x0;
        if (!machineIsBigEndian) FlipEndianness(&header_length_write);
        if (!machineIsBigEndian) FlipEndianness(&output_width);
        if (!machineIsBigEndian) FlipEndianness(&output_height);
        std::memcpy(header, "IHDR", 4);
        std::memcpy(header+4, &output_width, 4);
        std::memcpy(header+8, &output_height, 4);
        std::memcpy(header+12, &bit_depth, 1);
        std::memcpy(header+13, &color_type, 1);
        std::memcpy(header+14, &compression_method, 1);
        std::memcpy(header+15, &filter_method, 1);
        std::memcpy(header+16, &interlace_method, 1);
        fwrite(&header_length_write, 1, sizeof(int), fileWriter);
        fwrite(header, 1, (4+header_length)*sizeof(ubyte), fileWriter);

        int checksumval = checksum.ComputeChecksum(header, header_length+4);
        if (!machineIsBigEndian) FlipEndianness(&checksumval);
        fwrite(&checksumval, 1, sizeof(int), fileWriter);
    }
    void PngImage::WriteEnd(FILE* fileWriter)
    {
        ubyte endblock[12] = {0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
        fwrite(endblock, 1, 12*sizeof(ubyte), fileWriter);
    }
    void PngImage::FlipEndianness(int* input)
    {
        int j;
        ubyte byte_value;
        for (j = 0; j < sizeof(int)/2; j++)
        {
            byte_value = *(((ubyte*)input)+j);
            *(((ubyte*)input)+j) = *(((ubyte*)input)+sizeof(int)-j-1);
            *(((ubyte*)input)+sizeof(int)-j-1) = byte_value;
        }
    }
}
