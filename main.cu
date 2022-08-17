#include <cstdio>
#include <cuda_runtime.h>
#include "CudaAllocator.h"
#include "helper_cuda.h"
#include <cmath>
#include <vector>
// #include <thrust/device_vector.h>  // 如果想用 thrust 也是没问题的
#include <thrust/universal_vector.h>
#include <ticktock.h>

// 这是基于“边角料法”的，请把他改成基于“网格跨步循环”的：10 分
template <class Fun>
__global__ void fill_sin(int n, Fun func) {
    int step = gridDim.x * blockDim.x;
    for(size_t i=blockIdx.x * blockDim.x + threadIdx.x; i < n; i+=step)
        func(i);
}

template <int N, class T1, class T2, class T3>
 __global__ void filter_positive(T1 arr, T2 res, T3 cnt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > N) return;
    if(arr[i] >= 0){
        int loc = atomicAdd(&cnt[0], 1);
        res[loc] = arr[i];
    }
}

int main() {
    constexpr int n = 1<<24;
    thrust::universal_vector<float> arr(n);
    thrust::universal_vector<float> res(n);
    thrust::universal_vector<int> counter(1);

    // TICK(FILL_SIN);
    fill_sin<<<n / 1024, 32>>>(n, 
    [arr = arr.data()] __device__ (size_t i){
        arr[i] = sinf(i);
    });
    // checkCudaErrors(cudaDeviceSynchronize());
    // TOCK(FILL_SIN);

    TICK(filter_positive);
    filter_positive<n><<<(n + 1023) / 1024, 1024>>>(arr.data(), res.data(), counter.data());
    // checkCudaErrors(cudaDeviceSynchronize());
    TOCK(filter_positive);

    checkCudaErrors(cudaDeviceSynchronize());
    if (counter[0] <= n / 50) {
        printf("Result too short! %d <= %d\n", counter[0], n / 50);
        return -1;
    }
    for (int i = 0; i < counter[0]; i++) {
        if (res[i] < 0) {
            printf("Wrong At %d: %f < 0\n", i, res[i]);
            return -1;  // 突然想起了ICPC有一年队名叫“蓝翔WA掘机”的，笑不活了:)
        }
    }

    printf("All Correct!\n");  // 还有个队名叫“AC自动机”的，和隔壁“WAWA大哭”对标是吧:)
    return 0;
}
