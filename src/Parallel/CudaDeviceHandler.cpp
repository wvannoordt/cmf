#include "CudaDeviceHandler.h"
#include "CmfScreen.h"
#include "StringUtils.h"
namespace cmf
{
    CudaDeviceHandler::CudaDeviceHandler(void)
    {
        numberOfDevices = 0;
        CMF_CUDA_CHECK(cudaGetDeviceCount(&numberOfDevices));
        WriteLine(2, "Found " + std::to_string(numberOfDevices) + " CUDA-enabled device" + ((numberOfDevices==1)?(""):("s")));
        for (int i = 0; i < numberOfDevices; i++)
        {
            CmfCudaDeviceProperties devProps;
            CMF_CUDA_CHECK(cudaGetDeviceProperties(&devProps, i));
            deviceProperties.push_back(devProps);
            PrintProperties(i, devProps);
        }
    }
    
    void CudaDeviceHandler::PrintProperties(int i, CmfCudaDeviceProperties& devProps)
    {
        std::string style = " >> ";
        WriteLine(3, "-------------------------------------------------------");
        WriteLine(3, "CUDA device " + std::to_string(i) + ":");
        WriteLine(3, style + "Name:                     " + devProps.name);
        WriteLine(3, style + "Global memory:            " + UnitString(devProps.totalGlobalMem, "B", 1024));
        WriteLine(3, style + "Shared memory per block:  " + UnitString(devProps.sharedMemPerBlock, "B", 1024));
        WriteLine(3, style + "Warp size:                " + std::to_string(devProps.warpSize));
        WriteLine(3, style + "Max threads per block:    " + std::to_string(devProps.maxThreadsPerBlock));
        WriteLine(3, style + "Max thread dimension:     "
            + NiceCommaString(devProps.maxThreadsDim[0]) + "  x  "
            + NiceCommaString(devProps.maxThreadsDim[1]) + "  x  "
            + NiceCommaString(devProps.maxThreadsDim[2]));
        WriteLine(3, style + "Max grid size:            "
            + NiceCommaString(devProps.maxGridSize[0]) + "  x  "
            + NiceCommaString(devProps.maxGridSize[1]) + "  x  "
            + NiceCommaString(devProps.maxGridSize[2]));
        WriteLine(3, style + "Total constant memory:    " + UnitString(devProps.totalConstMem, "B", 1024));
        WriteLine(3, style + "Compute capability:       " + std::to_string(devProps.major) + "." + std::to_string(devProps.minor));
        WriteLine(3, style + "Clock rate:               " + UnitString(devProps.clockRate, "Hz"));
        WriteLine(3, style + "Multiprocessor count:     " + std::to_string(devProps.multiProcessorCount));
        WriteLine(3, style + "Device PCI bus/device ID: " + std::to_string(devProps.pciBusID) + "/" + std::to_string(devProps.pciDeviceID));
        WriteLine(3, "-------------------------------------------------------");
    }
    
    CudaDeviceHandler::~CudaDeviceHandler(void)
    {
        
    }
}