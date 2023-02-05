#include <cstdio>
#include <cuda_runtime.h>
#include "CudaAllocator.h"
#include "helper_cuda.h"
#include <cmath>
#include <vector>
 #include <thrust/device_vector.h>  // 如果想用 thrust 也是没问题的



// 这是基于“边角料法”的，请把他改成基于“网格跨步循环”的：10 分
template<class Func>
__global__ void parallel_for(int n, Func func){
    for(int i = blockDim.x * blockIdx.x + threadIdx.x;
        i < n ; i += blockDim.x * gridDim.x){
        func(i);
    }
}


//原方法问题，多线程对同时读取/修改/写回counter;
//采取原子操作解决；

__global__ void filter_positive(int *counter, int *res, int const *arr, int n){
    for(int i = blockDim.x * blockIdx.x + threadIdx.x;
        i < n ; i += blockDim.x * gridDim.x){
        if(arr[i] >= 0 ){
            int loc = atomicAdd(counter,1);
            res[loc] = n;
        }
    }
}


int main() {
    constexpr int n = 1<<24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> res(n);
    std::vector<int, CudaAllocator<int>> counter(1);

    // fill_sin 改成“网格跨步循环”以后，这里三重尖括号里的参数如何调整？10 分
    parallel_for<<<32,1014>>>(n,[arr_data = arr.data()] __device__ (int i ){
        arr_data[i] = sinf(i);
    });
    
    // 这里的“边角料法”对于不是 1024 整数倍的 n 会出错，为什么？请修复：10 分
    //原因：如果不是整数倍，就会漏掉最后几个元素；既可以采用向上取整，也可以网格跨步循环来解决；
    filter_positive<<<32,1024>>>(counter.data(), res.data(), arr.data(), n);


    // 这里 CPU 访问数据前漏了一步什么操作？请补上：10 分
    //调用 cudaDeviceSynchronize()，让 CPU 陷入等待，等 GPU 完成队列的所有任务后再返回。
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
