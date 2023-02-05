#include <cstdio>
#include <cuda_runtime.h>
#include "CudaAllocator.h"
#include "helper_cuda.h"
#include <cmath>
#include <vector>
// #include <thrust/device_vector.h>  // 如果想用 thrust 也是没问题的

// 这是基于“边角料法”的，请把他改成基于“网格跨步循环”的：10 分
// 总的线程数量：blockDim * gridDim
// 总的线程编号：blockDim * blockIdx + threadIdx
// 网格跨步循环利用扁平化的线程数量和线程编号实现动态大小。
// for 循环非常符合 CPU 上常见的 parallel for 的习惯，又能自动匹配不同的 blockDim 和 gridDim
// 核函数可以是一个模板函数
template<class Func>
__global__ void parallel_for(int n, Func func){
    for(int i = blockDim.x * blockIdx.x + threadIdx.x;i < n ; i += blockDim.x * gridDim.x){
        func(i);
    }
}

__global__ void filter_positive(int *counter, int *res, int const *arr, int n) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n ; i += blockDim.x * gridDim.x){
        if(arr[i] >= 0 ){
            int loc = atomicAdd(counter,1);
            res[loc] = n;
        }
    }
}

// vector 会调用 std::allocator<T> 的 allocate/deallocate 成员函数，他又会去调用标准库的 malloc/free 分配和释放内存空间
// 我们可以自己定义一个和 std::allocator<T> 一样具有 allocate/deallocate 成员函数的类，这样就可以“骗过”vector，让他不是在 CPU 内存中分配，而是在 CUDA 的统一内存(managed)上分配。
int main() {
    constexpr int n = 1<<24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> res(n);
    std::vector<int, CudaAllocator<int>> counter(1);

    // fill_sin 改成“网格跨步循环”以后，这里三重尖括号里的参数如何调整？10 分
    // 在 [] 里这样直接写自定义捕获的表达式也是可以的，这样就可以用同一变量名
    // 两个下划线的 __sinf 是 GPU intrinstics，精度相当于 GLSL 里的那种。适合对精度要求不高，但有性能要求的图形学任务
    parallel_for<<<32,1024>>>(n,[arr_data = arr.data()] __device__ (int i){
        arr_data[i] = __sinf(i);
    });

    // 这里的“边角料法”对于不是 1024 整数倍的 n 会出错，为什么？请修复：10 分
    // answer: 由于向上取整，这样会多出来一些线程，因此要在 kernel 内判断当前 i 是否超过了 n，如果超过就要提前退出，防止越界
    filter_positive<<< (n+1024-1) / 1024, 1024>>>(counter.data(), res.data(), arr.data(), n);

    // 这里 CPU 访问数据前漏了一步什么操作？请补上：10 分
    // answer:可以调用 cudaDeviceSynchronize()，让 CPU 陷入等待，等 GPU 完成队列的所有任务后再返回。
    // 从而能够在 main 退出前等到 kernel 在 GPU 上执行完
    // checkCudaErrors 可自动帮你检查错误代码并打印在终端，然后退出。还会报告出错所在的行号，函数名等
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
