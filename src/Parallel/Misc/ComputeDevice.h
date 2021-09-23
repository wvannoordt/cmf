#ifndef CMF_COMPUTE_DEVICE_H
#define CMF_COMPUTE_DEVICE_H
#include <iostream>
namespace cmf
{
    ///@brief Represents a generic computing device, including a CPU computing node, or a Nvidia GPU
    ///@author WVN
    struct ComputeDevice
    {
        ///@brief indicates if the device is a GPU or not (currently only supporting Nvidia GPUs)
        bool isGpu;
        
        ///@brief if this device is a gpu, deviceNum indicates its device identifier relative to its host
        int deviceNum;
        
        ///@brief an ID code that delimits the rank of this device in its pool. Note that if
        ///this device is a GPU, then this id indicates the rank of its host in the relevant parallel
        ///group
        int id;
        
        ///@brief Constructor
        ///@author WVN
        ComputeDevice(bool isGpu_in, int id_in)
        {
            isGpu = isGpu_in;
            id = id_in;
            deviceNum = 0;
        }
        
        ///@brief Constructor
        ///@author WVN
        ComputeDevice(bool isGpu_in, int id_in, int deviceNum_in)
        {
            isGpu = isGpu_in;
            id = id_in;
            deviceNum = deviceNum_in;
        }
        
        ///@brief Constructor
        ///@author WVN
        ComputeDevice(void)
        {
            isGpu = false;
            deviceNum = 0;
            id = 0;
        }
        
        ///@brief Equality operator
        ///@param rhs the device to test equality to
        ///@author WVN
        bool operator == (const ComputeDevice& rhs)
        {
            return (isGpu==rhs.isGpu) && (rhs.id==id);
        }
        
        ///@brief Inquality operator
        ///@param rhs the device to test equality to
        ///@author WVN
        bool operator != (const ComputeDevice& rhs)
        {
            return (isGpu!=rhs.isGpu) || (rhs.id!=id);
        }
        
    };
    
    ///@brief Allows streaming of ComputeDevice
    static std::ostream & operator<<(std::ostream & os, const ComputeDevice & dev)
    {
        os << "{id:" << dev.id << ", devNum:" << dev.deviceNum << ", gpu:" << (dev.isGpu?"true":"false") << "}";
        return os;
    }
}

#endif