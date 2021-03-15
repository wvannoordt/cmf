#ifndef CMF_CUDA_DEVICE_HANDLER_H
#define CMF_CUDA_DEVICE_HANDLER_H
#include "CmfCuda.h"
#include "Typedefs.h"
#include <vector>
namespace cmf
{
    /// @brief Handles details of setting active CUDA device,device properties, etc.
	/// @author WVN
    class CudaDeviceHandler
    {
        public:
            /// @brief Constructor. Gets information on all available CUDA devices (current node only)
        	/// @author WVN
            CudaDeviceHandler(void);
            /// @brief Destructor
        	/// @author WVN
            ~CudaDeviceHandler(void);
            
        private:
            /// @brief The number of devices available to the current node
            int numberOfDevices;
            
            /// @brief The properties of each CUDA device
            std::vector<CmfCudaDeviceProperties> deviceProperties;
            
            /// @brief Prints a CmfCudaDeviceProperties struct to the screen
            /// @author WVN
            /// @param i the device to print properties for
            /// @param devProps the device properties to print
            void PrintProperties(cudaDev_t i, CmfCudaDeviceProperties& devProps);
    };
}

#endif