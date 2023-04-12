#include <cstdio>
#include <cuda_runtime.h>
#include "CudaAllocator.h"
#include "helper_cuda.h"
#include <cmath>
#include <vector>
#include "ticktock.h"
// #include <thrust/device_vector.h>  // 如果想用 thrust 也是没问题的

// 这是基于“边角料法”的，请把他改成基于“网格跨步循环”的：10 分
__global__ void fill_sin(float *arr, int n) {
    for (int i= threadIdx.x+ blockIdx.x*  blockDim.x;i<n;i+= gridDim.x* blockDim.x){
        arr[i]=sinf(i);
        // printf("%f\n",arr[i]);
    }
}

template<int N,int blockSize>
__global__ void pre_exclusive_scan(float const *arr,int *writable,int *wIndex,int *aux) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid > N) return;
    int tid=threadIdx.x;
    __shared__ int prefix[blockSize*2];
    prefix[tid*2]=writable[gid*2]=arr[gid*2]>=0?1:0;
    prefix[tid*2+1]=writable[gid*2+1]=arr[gid*2+1]>=0?1:0;
    // printf("%d,%d\n",writable[gid*2],writable[gid*2+1]);
    __syncthreads();
    //inclusive_scan
    // for(int stride=1;stride<=tid;stride*=2){
    //     prefix[tid]+=prefix[tid-stride];
    //     __syncthreads();
    // }
    //exclusive_scan
    for (int stride=1;stride<=blockSize;stride*=2){
        int index=(tid+1)*stride*2-1;
        if (index<blockSize*2)
            prefix[index]+=prefix[index-stride];
        __syncthreads();
    }
    if (tid==0)
        prefix[blockSize*2-1]=0;
    int tmp=0;
    for(int stride=blockSize;stride>0;stride/=2){
        int index=(tid+1)*stride*2-1;
        if (index < blockSize*2){
            tmp=prefix[index];
            prefix[index]+=prefix[index-stride];
            prefix[index-stride]=tmp;
        }
        __syncthreads();
    }
    wIndex[gid*2]=prefix[tid*2];
    wIndex[gid*2+1]=prefix[tid*2+1];
    if(tid==0){
        aux[blockIdx.x]=prefix[blockSize*2-1]+writable[gid*2+1];
        // printf("%d\n",aux[blockIdx.x]);
    }
}

template<int blockSize>
__global__ void single_block_exclusive_scan(int *aux){
    int tid = threadIdx.x;
    __shared__ int local_aux[blockSize*2];
    local_aux[tid*2]=aux[tid*2];
    local_aux[tid*2+1]=aux[tid*2+1];
    __syncthreads();
    for (int stride=1;stride<=blockSize;stride*=2){
        int index=(tid+1)*stride*2-1;
        if (index<blockSize*2)
            local_aux[index]+=local_aux[index-stride];
        __syncthreads();
    }
    if (tid==0)
        local_aux[blockSize*2-1]=0;
    int tmp=0;
    for(int stride=blockSize;stride>0;stride/=2){
        int index=(tid+1)*stride*2-1;
        if (index < blockSize*2){
            tmp=local_aux[index];
            local_aux[index]+=local_aux[index-stride];
            local_aux[index-stride]=tmp;
        }
        __syncthreads();
    }
    aux[tid*2]=local_aux[tid*2];
    aux[tid*2+1]=local_aux[tid*2+1];
    printf("%d,%d\n",aux[tid*2],aux[tid*2+1]);
}

template<int N,int blockSize>
__global__ void post_scan(float const *arr, int *aux,int *wIndex, float *res,int *writable){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid > N) return;
    wIndex[gid*2]+=aux[blockIdx.x];
    wIndex[gid*2+1]+=aux[blockIdx.x];
    if (writable[gid*2])
        res[wIndex[gid*2]]=arr[gid*2];
    if (writable[gid*2+1])
        res[wIndex[gid*2+1]]=arr[gid*2+1];
}

template <int N,int blockSize>
void filter_positive(int *counter, float *res, float const *arr){
    std::vector<int, CudaAllocator<int>> writable(N);
    std::vector<int, CudaAllocator<int>> wIndex(N);
    std::vector<int, CudaAllocator<int>> aux(N/2/blockSize);
    pre_exclusive_scan<N,blockSize><<<N/2/blockSize,blockSize>>>(arr,writable.data(),wIndex.data(),aux.data());
    single_block_exclusive_scan<N/4/blockSize><<<1,N/4/blockSize>>>(aux.data());
    post_scan<N,blockSize><<<N/2/blockSize,blockSize>>>(arr,aux.data(),wIndex.data(),res,writable.data());
    cudaDeviceSynchronize();
    counter[0]=wIndex.back();
}

int main() {
    constexpr int n = 1<<24;
    std::vector<float, CudaAllocator<float>> arr(n);
    std::vector<float, CudaAllocator<float>> res(n);
    std::vector<int, CudaAllocator<int>> counter(1);

    TICK(kernel);

    // fill_sin 改成“网格跨步循环”以后，这里三重尖括号里的参数如何调整？10 分
    fill_sin<<<n / 32 /1024, 1024>>>(arr.data(), n);

    // 这里的“边角料法”对于不是 1024 整数倍的 n 会出错，为什么？请修复：10 分
    // filter_positive<n,1024><<< (n+1023) / 1024, 1024>>>(counter.data(), res.data(), arr.data());
    filter_positive<n,1024>(counter.data(), res.data(), arr.data());
    // 这里 CPU 访问数据前漏了一步什么操作？请补上：10 分

    TOCK(kernel);

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
