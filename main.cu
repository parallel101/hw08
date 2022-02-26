#include <cstdio>
#include <cuda_runtime.h>
// #include "CudaAllocator.h"
#include "helper_cuda.h"
#include <cmath>
// #include <vector>
// #include <thrust/device_vector.h>  // 如果想用 thrust 也是没问题的
#define DEBUG_WRONG_ANS 0;

// 1. 这是基于“边角料法”的，请把他改成基于“网格跨步循环”的：10 分
__global__ void fill_sin(int* arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        arr[i] = sinf(i);
    }
}

__global__ void filter_positive(int* counter, int* res, int const* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (arr[i] >= 0) {
        // 2. 这里有什么问题？请改正：10 分
        // 答：应改为原子操作
        int loc = atomicAdd(counter, 1);
        res[loc] = n;
    }
}

int main() {
    constexpr int n = 1 << 24;
    // std::vector<int, CudaAllocator<int>> arr(n);
    // std::vector<int, CudaAllocator<int>> res(n);
    // std::vector<int, CudaAllocator<int>> counter(1);

    int* arr;
    int* res;
    int* counter;
    checkCudaErrors(cudaMalloc(&arr, n * sizeof(int)));
    checkCudaErrors(cudaMalloc(&res, n * sizeof(int)));
    checkCudaErrors(cudaMalloc(&counter, sizeof(int)));

    // 3. fill_sin 改成“网格跨步循环”以后，这里三重尖括号里的参数如何调整？10 分
    // 答：改为 grid-stide loop 后，就不用保证 gridDim * blockDim == n 啦！
    fill_sin << <32, 1024 >> > (arr, n);

    // 4. 这里的“边角料法”对于不是 1024 整数倍的 n 会出错，为什么？请修复：10 分
    // 答：由于直接作除法会向下取整，可能遗漏数据。应改为向上取整，额外分配一个 block 来处理末尾数据
    int grid_dim = (n + 1023) / 1024;
    int block_dim = 1024;
    filter_positive << <grid_dim, block_dim >> > (counter, res, arr, n);

    // 5. 这里 CPU 访问数据前漏了一步什么操作？请补上：10 分
    // 答：由于 CPU 和 GPU 是异步执行的，故在 CPU 访问 GPU 计算好的数据前应该先执行同步
    checkCudaErrors(cudaDeviceSynchronize());
    
    int* res_host;  // 把数据拷贝回 CPU 以访问
    int* counter_host;
    
    res_host = (int *)malloc(n * sizeof(int));
    counter_host = (int *)malloc(sizeof(int));
    
    checkCudaErrors(cudaMemcpy(res_host, res, n * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(counter_host, counter, sizeof(int), cudaMemcpyDeviceToHost));
#if DEBUG_WRONG_ANS
    res_host[0] = -1;  // 写入一个负值测试
#endif
    if (counter_host[0] <= n / 50) {
        printf("Result too short! %d <= %d\n", counter_host[0], n / 50);
        return -1;
    }
    for (int i = 0; i < counter_host[0]; i++) {
        if (res_host[i] < 0) {
            printf("Wrong At %d: %d < 0\n", i, res_host[i]);
            return -1;  // 突然想起了ICPC有一年队名叫“蓝翔WA掘机”的，笑不活了:)
        }
    }
    printf("All Correct!\n");  // 还有个队名叫“AC自动机”的，和隔壁“WAWA大哭”对标是吧:)
    cudaFree(arr);
    cudaFree(res);
    cudaFree(counter);
    free(res_host);
    free(counter_host);
    return 0;
}

