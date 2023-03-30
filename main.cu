#include <cstdio>
#include <cuda_runtime.h>
#include "CudaAllocator.h"
#include "helper_cuda.h"
#include <cmath>
#include <vector>
// #include <thrust/device_vector.h>  // 如果想用 thrust 也是没问题的

// 这是基于“边角料法”的，请把他改成基于“网格跨步循环”的：10 分
__global__ void fill_sin(int *arr, int n) {
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i=blockIdx.x * blockDim.x + threadIdx.x; i<n; i+=blockDim.x*gridDim.x){
        arr[i] = sinf(i);
    }
    
}

__global__ void filter_positive(int *counter, int *res, int const *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n) return;
    if (arr[i] >= 0) {
        // 这里有什么问题？请改正：10 分
        // ans: 多线程会同时访问和修改counter，需要使用原子操作。
        // int loc = *counter;
        // atomicAdd(&)
        // *counter += 1;
        // res[loc] = n;

        int loc = atomicAdd(counter, 1);
        res[loc] = n;
        // printf("%d\n", loc);
    }
}

int main() {
    constexpr int n = 1<<24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> res(n);
    std::vector<int, CudaAllocator<int>> counter(1);

    // fill_sin 改成“网格跨步循环”以后，这里三重尖括号里的参数如何调整？10 分
    // ans: 将block数调整为32即可。
    fill_sin<<<32, 1024>>>(arr.data(), n);

    // 这里的“边角料法”对于不是 1024 整数倍的 n 会出错，为什么？请修复：10 分
    // ans: 会缺少数据，应该向上取整，让总线程数大于n。
    filter_positive<<<(n+1024-1) / 1024, 1024>>>(counter.data(), res.data(), arr.data(), n);
    // cudaDeviceSynchronize();

    // 这里 CPU 访问数据前漏了一步什么操作？请补上：10 分
    // ans: 要将数据从cuda拷贝到内存上。
    int counter_cpu;
    std::vector<int> res_cpu(n);
    checkCudaErrors(cudaMemcpy(&counter_cpu, counter.data(), sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(res_cpu.data(), res.data(), sizeof(int)*res.size(), cudaMemcpyDeviceToHost));

    if (counter_cpu <= n / 50) {
        printf("Result too short! %d <= %d\n", counter_cpu, n / 50);
        return -1;
    }
    for (int i = 0; i < counter_cpu; i++) {
        if (res[i] != n) {
            printf("Wrong At %d: %d < 0\n", i, res[i]);
            return -1;  // 突然想起了ICPC有一年队名叫“蓝翔WA掘机”的，笑不活了:)
        }
    }

    printf("All Correct!\n");  // 还有个队名叫“AC自动机”的，和隔壁“WAWA大哭”对标是吧:)
    return 0;
}
