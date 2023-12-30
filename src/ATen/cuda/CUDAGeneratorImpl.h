#pragma once

namespace at
{
    /* Since this isn't meant for PyTorch, the state is always a pair of concrete ints
       instead of an untagged union. 
       May not work with PyTorch JIT, but this isn't meant to be used with PyTorch
    */
    struct PhiloxCudaState
    {
        unsigned long long seed_;
        unsigned long long offset_;
        PhiloxCudaState() = default; 
        PhiloxCudaState(unsigned long long seed, unsigned long long offset):
            seed_(seed),
            offset_(offset) {};
    };
}