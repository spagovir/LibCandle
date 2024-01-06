#pragma once

#include <string>
#include <stdexcept>

#define C10_CUDA_CHECK(expr) libcandle::check(expr, __FILE__, __func__, static_cast<int>(__LINE__))
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

namespace at::cuda
{
    cudaDeviceProp* getDeviceProperties(int64_t device)
    {
        cudaDeviceProp* prop = new cudaDeviceProp;
        C10_CUDA_CHECK(cudaGetDeviceProperties(prop, device))
        return prop;
    }
    cudaDeviceProp* getCurrentDeviceProperties()
    {
        int device;
        c10_CUDA_CHECK(cudaGetDevice(&device));
        return device;
    }
        
}

namespace libcandle 
{ 
    void check(cudaError_t err, char* file, char* function, int line)
    {
        if(err == cudaSuccess)
            return;
        else 
        {
            std::string err_msg = "CUDA Error: ";
            err_msg.append(cudaGetErrorString(err));
            err_msg.append(" at \n\tfile: ");
            err_msg.append(file);
            err_msg.append("\n\tfunction: ");
            err_msg.append(function);
            err_msg.append("\n\tline: ");
            err_msg.append(std::to_string(line));
            err_msg.append("\n");
            throw(std::runtime_error(err_msg));
        }
    }
}
