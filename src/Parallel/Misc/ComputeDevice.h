#ifndef CMF_COMPUTE_DEVICE_H
#define CMF_COMPUTE_DEVICE_H

namespace cmf
{
    ///@brief Represents a generic computing device, including a CPU computing node, or a Nvidia GPU
    ///@author WVN
    struct ComputeDevice
    {
        ///@brief indicates if the device is a GPU or not (currently only supporting Nvidia GPUs)
        bool isGpu;
        
        ///@brief an ID code that delimits the rank of this device in its pool
        int id;
        
        ///@brief Constructor
        ///@author WVN
        ComputeDevice(bool isGpu_in, int id_in)
        {
            isGpu = isGpu_in;
            id = id_in;
        }
        
        ///@brief Constructor
        ///@author WVN
        ComputeDevice(void)
        {
            isGpu = false;
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
}

#endif