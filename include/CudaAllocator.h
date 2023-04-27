#pragma once

#include <cstddef>
#include <utility>
#include <cuda_runtime.h>
#include "helper_cuda.h"

template <class T>
struct CudaAllocator {
    using value_type = T;

    T *allocate(size_t size) {
        T *ptr = nullptr;
        checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
        return ptr;
    }

    void deallocate(T *ptr, size_t size = 0) {
        checkCudaErrors(cudaFree(ptr));
    }

    template <class ...Args>
    void construct(T *p, Args &&...args) {
        if (!(sizeof...(Args) == 0 ))
            ::new((void *)p) T(std::forward<Args>(args)...);
    }
    
    // //**** CIHOU SHABI WENDOUS ****
    // template<class _Other>
    // constexpr CudaAllocator(const CudaAllocator<_Other>&) noexcept {}

    // constexpr bool operator==(CudaAllocator<T> const &other) const {
    //     return this == &other;
    // }
};
