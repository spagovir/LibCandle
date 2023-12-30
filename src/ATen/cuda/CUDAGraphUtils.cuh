#pragma once
#include <tuple>
#include "CUDAGeneratorImpl.h"
namespace at::cuda::philox 
{
    __host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t>
    unpack(at::PhiloxCudaState arg) {
        return std::make_tuple(arg.seed_, arg.offset_);
    }
}