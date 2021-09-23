#ifndef CMF_CUDA_INCL_H
#define CMF_CUDA_INCL_H
#include "ConfigParams.h"
#include "CmfError.h"
//Enables usage of CUDA kernels
#ifndef CUDA_ENABLE
#define CUDA_ENABLE 0
#endif

#if(CUDA_ENABLE)
#include <cuda.h>
#include "cuda_device_runtime_api.h"
#include "cuda_runtime_api.h"
#include <cuda_runtime.h>
#define CMF_CUDA_CHECK(myCudaCall) {cudaError_t er_INTERNAL = (myCudaCall); if (er_INTERNAL!=cudaSuccess) {CmfError(std::string("F") + "ailed call to CUDA runtime API. Call: " + #myCudaCall + "\nError: " + cudaGetErrorString(er_INTERNAL));}}
typedef cudaDeviceProp CmfCudaDeviceProperties;
#else
#define CMF_CUDA_CHECK(myCudaCall) ;
#endif

namespace cmf
{
#if(!CUDA_ENABLE)
    /// @brief Wraps cudaDeviceProp (see CUDA documentation)
    /// @author WVN
    struct CmfCudaDeviceProperties
    {
        /// @brief an ASCII string identifying the device
        char name[256];
        
        /// @brief the total amount of global memory available on the device in bytes
        size_t totalGlobalMem;
        
        /// @brief the maximum amount of shared memory available to a thread block in bytes; this amount is shared by all thread blocks simultaneously resident on a multiprocessor
        size_t sharedMemPerBlock;
        
        /// @brief the maximum number of 32-bit registers available to a thread block; this number is shared by all thread blocks simultaneously resident on a multiprocessor
        int regsPerBlock;
        
        /// @brief the warp size in threads
        int warpSize;
        
        /// @brief the maximum pitch in bytes allowed by the memory copy functions that involve memory regions allocated through cudaMallocPitch()
        size_t memPitch;
        
        /// @brief the maximum number of threads per block
        int maxThreadsPerBlock;
        
        /// @brief contains the maximum size of each dimension of a block
        int maxThreadsDim[3];
        
        /// @brief contains the maximum size of each dimension of a grid
        int maxGridSize[3];
        
        /// @brief is the total amount of constant memory available on the device in bytes
        size_t totalConstMem;
        
        /// @brief major revision numbers defining the device's compute capability
        int major;
        
        /// @brief minor revision numbers defining the device's compute capability
        int minor;
        
        /// @brief is the clock frequency in kilohertz
        int clockRate;
        
        /// @brief the alignment requirement; texture base addresses that are aligned to textureAlignment bytes do not need an offset applied to texture fetches
        size_t textureAlignment;
        
        /// @brief 1 if the device can concurrently copy memory between host and device while executing a kernel, or 0 if not
        int deviceOverlap;
        
        /// @brief the number of multiprocessors on the device
        int multiProcessorCount;
        
        /// @brief 1 if there is a run time limit for kernels executed on the device, or 0 if not
        int kernelExecTimeoutEnabled;
        
        /// @brief 1 if the device is an integrated (motherboard) GPU and 0 if it is a discrete (card) component
        int integrated;
        
        /// @brief 1 if the device can map host memory into the CUDA address space for use with cudaHostAlloc()/cudaHostGetDevicePointer(), or 0 if not
        int canMapHostMemory;
        
        /// @brief the compute mode that the device is currently in
        int computeMode;
        
        /// @brief 1 if the device supports executing multiple kernels within the same context simultaneously, or 0 if not.
        /// It is not guaranteed that multiple kernels will be resident on the device concurrently so this feature should
        /// not be relied upon for correctness
        int concurrentKernels;
        
        /// @brief 1 if the device has ECC support turned on, or 0 if not
        int ECCEnabled;
        
        /// @brief the PCI bus identifier of the device
        int pciBusID;
        
        /// @brief the PCI device (sometimes called slot) identifier of the device
        int pciDeviceID;
        
        /// @brief 1 if the device is using a TCC driver or 0 if not
        int tccDriver;
    };
#endif
    
    namespace DeviceTransferDirection
    {
        enum DeviceTransferDirection
        {
            GpuToCpu,
            CpuToGpu,
            GpuToGpu
        };
    }
    
    /// @brief Wrapper for cudaMemcpy
    /// @param source the data source
    /// @param destination the data destination
    /// @param size the size (bytes) of the data to copy
    /// @param gpuId the device to Transfer to (or from)
    /// @param direction the direction to copy (one of DeviceTransferDirection::GpuToCpu or DeviceTransferDirection::CpuToGpu)
    /// @author WVN
    template <const DeviceTransferDirection::DeviceTransferDirection direction> static inline void GpuMemTransfer(void* source, void* destination, size_t size, int gpuId)
    {
        CMF_CUDA_CHECK(cudaSetDevice(gpuId));
        switch (direction)
        {
            case DeviceTransferDirection::GpuToCpu:
            {
                CMF_CUDA_CHECK(cudaMemcpy(destination, source, size, cudaMemcpyDeviceToHost));
                break;
            }
            case DeviceTransferDirection::CpuToGpu:
            {
                CMF_CUDA_CHECK(cudaMemcpy(destination, source, size, cudaMemcpyHostToDevice));
                break;
            }
            case DeviceTransferDirection::GpuToGpu:
            {
                CMF_CUDA_CHECK(cudaMemcpy(destination, source, size, cudaMemcpyDeviceToDevice));
                break;
            }
        }
    }
}

#endif