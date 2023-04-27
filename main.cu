#include <cstdio>
#include <cuda_runtime.h>
#include "CudaAllocator.h"
#include "helper_cuda.h"
#include <cmath>
#include <vector>
#include "ticktock.h"
#include "kernel.cuh"
// #include <thrust/device_vector.h>  // 如果想用 thrust 也是没问题的

#define ELEMENTS_PER_BLOCK (thread_per_block*2)

template<class T,int thread_per_block>
void scan(T *d_in,T *d_out,int length){
    if(length<=ELEMENTS_PER_BLOCK)
        singleBlockScan<T><<<1,thread_per_block>>>(d_in,d_out,length);
    else{
        largeDataScan<T>(d_in,d_out,length);
    }
}

template<class T,int thread_per_block>
__global__ void singleBlockScan(T *d_in,T *d_out,int length){
    int tid=threadIdx.x;
    __shared__ T bls_presum[ELEMENTS_PER_BLOCK];
    bls_presum[tid*2]=(tid*2)<length?d_in[tid*2]:0;
    bls_presum[tid*2+1]=(tid*2+1)<length?d_in[tid*2+1]:0;
    __syncthreads();
    for (int stride=1;stride<=thread_per_block;stride*=2){
        int index=(tid+1)*2*stride-1;
        if (index<ELEMENTS_PER_BLOCK)
            bls_presum[index]+=bls_presum[index-stride];
        __syncthreads();
    }
    if (tid==0){
        bls_presum[ELEMENTS_PER_BLOCK-1]=0;
    }
    int tmp;
    for (int stride=thread_per_block;stride>0;stride/=2){
        int index=(tid+1)*2*stride-1;
        if (index<ELEMENTS_PER_BLOCK){
            tmp=bls_presum[index];
            bls_presum[index]+=bls_presum[index-stride];
            bls_presum[index-stride]=tmp;
        }
        __syncthreads();
    }
    if (tid*2<length)
        d_out[tid*2]=bls_presum[tid*2];
    if (tid*2+1<length)
        d_out[tid*2+1]=bls_presum[tid*2+1];
}

template<class T,int thread_per_block>
void largeDataScan(T *d_in,T *d_out,int length){

    //计算超出ELEMENTS_PER_BLOCK倍数部分的元素数量
    int leftover=length%ELEMENTS_PER_BLOCK;

    //实际上preScan扫描的元素数量，剩余的部分使用singleBlockScan扫描计算
    int real_length=length-leftover;

    //cuda核函数网格维度
    int gridDim=real_length/ELEMENTS_PER_BLOCK;
    constexpr int blockDim=thread_per_block;

    //每个block内部元素的和，作为post_scan每个block内部的偏移量
    // std::vector<int, CudaAllocator<int>> block_offset(gridDim);
    int *dev_block_offset;
    checkCudaErrors(cudaMalloc((void **)&dev_block_offset,gridDim*sizeof(int)));

    //将large data分成一个个block*2(ELEMENTS_PER_BLOCK)大小的块，每个block内部进行scan
    preScan<T,blockDim><<<gridDim,blockDim>>>(d_in,d_out,dev_block_offset);
    //对block_offset进行exlusive_scan
    scan<T,blockDim>(dev_block_offset,dev_block_offset,gridDim);
    //将large data每个block局部scan的结果加上block_offset
    postScan<T,blockDim><<<gridDim*2,blockDim>>>(d_out,dev_block_offset);
    //同步等待上述核函数执行完毕
    checkCudaErrors(cudaDeviceSynchronize());
    //释放指针
    checkCudaErrors(cudaFree(dev_block_offset));

    //如果不是ELEMENTS_PER_BLOCK的整数倍，需要处理剩余的部分
    if(leftover){
        singleBlockScan<T,blockDim><<<1,blockDim>>>(&d_in[real_length],&d_out[real_length],leftover);
        //对d_out剩余部分进行
        procLeft<T><<<1,blockDim>>>(&d_out[real_length],&d_out[real_length-1],&d_in[real_length-1],leftover);
    }
}

template<class T,int thread_per_block>
__global__ void preScan(T *d_in,T *d_out,T *block_offset){
    int tid=threadIdx.x;
    int gid=threadIdx.x+thread_per_block*blockIdx.x;
    __shared__ T bls_presum[ELEMENTS_PER_BLOCK];
    bls_presum[tid*2]=d_in[gid*2];
    bls_presum[tid*2+1]=d_in[gid*2+1];
    __syncthreads();
    for (int stride=1;stride<=thread_per_block;stride*=2){
        int index=(tid+1)*2*stride-1;
        if (index<ELEMENTS_PER_BLOCK)
            bls_presum[index]+=bls_presum[index-stride];
        __syncthreads();
    }
    if (tid==0){
        block_offset[blockIdx.x]=bls_presum[ELEMENTS_PER_BLOCK-1];
        bls_presum[ELEMENTS_PER_BLOCK-1]=0;
    }
    T tmp;
    for (int stride=thread_per_block;stride>0;stride/=2){
        int index=(tid+1)*2*stride-1;
        if (index<ELEMENTS_PER_BLOCK){
            tmp=bls_presum[index];
            bls_presum[index]+=bls_presum[index-stride];
            bls_presum[index-stride]=tmp;
        }
        __syncthreads();
    }
    d_out[gid*2]=bls_presum[tid*2];
    d_out[gid*2+1]=bls_presum[tid*2+1];
}

template<class T,int thread_per_block>
__global__ void postScan(T *d_out,T *block_offset){
    int gid=threadIdx.x+thread_per_block*blockIdx.x;
    d_out[gid]+=block_offset[gid/ELEMENTS_PER_BLOCK];

}

template<class T>
__global__ void procLeft(T *d_out,T *num1,T *num2, int length){
    int tid=threadIdx.x;
    if(tid < length)
        d_out[tid]+=num1[0]+num2[0];
}

template<class T>
__global__ void filter(T *arr,int *isPos,int n){
    int gid=threadIdx.x+blockDim.x*blockIdx.x;
    if (gid>=n) return;
    isPos[gid]=(arr[gid]>=0?1:0);
}

template<class T>
__global__ void Compact(T *arr,T *res,int *isPos,int *index,int n){
    int gid=threadIdx.x+blockDim.x*blockIdx.x;
    if (gid>=n) return;
    if (isPos[gid])
        res[index[gid]]=arr[gid];
}

template<class T,int thread_per_block>
void filter_positive(int *counter,T *dev_res,T *dev_arr,int n){
    // std::vector<int, CudaAllocator<int>> isPos(n);
    // std::vector<int, CudaAllocator<int>> index(n);
    int *dev_isPos;
    int *dev_index;
    checkCudaErrors(cudaMalloc((void **)(&dev_isPos),n*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)(&dev_index),n*sizeof(int)));
    filter<T><<<(n+thread_per_block-1)/thread_per_block,thread_per_block>>>(dev_arr,dev_isPos,n);
    scan<int,thread_per_block>(dev_isPos,dev_index,n);
    Compact<T><<<(n+thread_per_block-1)/thread_per_block,thread_per_block>>>(dev_arr,dev_res,dev_isPos,dev_index,n);
    checkCudaErrors(cudaMemcpy(counter,&dev_index[n-1],sizeof(int),cudaMemcpyDeviceToHost));
}

// 这是基于“边角料法”的，请把他改成基于“网格跨步循环”的：10 分
__global__ void fill_sin(float *arr, int n) {
    for (int i= threadIdx.x+ blockIdx.x*  blockDim.x;i<n;i+= gridDim.x* blockDim.x){
        arr[i]=sinf(i);
        // printf("%f\n",arr[i]);
    }
}

__global__ void filter_positive_atomic_add(int *counter, float *res, float const *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n) return;
    if (arr[i] >= 0) {
        // 这里有什么问题？请改正：10 分
        int loc = atomicAdd(counter,1);
        res[loc]=arr[i];
    }
}

int main() {
    constexpr int n = 1<<24;//64MB数据
    std::vector<float, CudaAllocator<float>> arr(n);
    std::vector<float, CudaAllocator<float>> res(n);
    std::vector<int, CudaAllocator<int>> counter_ad(1);

    //===========直接在核函数内部使用原子指令对index进行递增============
    TICK(atomic_add);
    fill_sin<<<n / 32 /1024, 1024>>>(arr.data(), n);
    filter_positive_atomic_add<<< (n+1023) / 1024, 1024>>>(counter_ad.data(), res.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(atomic_add);

    float *dev_arr;
    float *dev_res;
    float *h_res=(float *)malloc(n*sizeof(float));
    int counter[1];

    //===========手写exclusive_scan_and_compact方法进行filter===========
    TICK(exclusive_scan_and_compact);
    checkCudaErrors(cudaMalloc((void **)(&dev_arr),n*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)(&dev_res),n*sizeof(float)));
    // fill_sin 改成“网格跨步循环”以后，这里三重尖括号里的参数如何调整？10 分
    fill_sin<<<n / 32 /1024, 1024>>>(dev_arr, n);

    // 这里的“边角料法”对于不是 1024 整数倍的 n 会出错，为什么？请修复：10 分
    // filter_positive<n,1024><<< (n+1023) / 1024, 1024>>>(counter.data(), res.data(), dev_arr);
    filter_positive<float,THREADS_PER_BLOCK>(counter, dev_res, dev_arr,n);
    // 这里 CPU 访问数据前漏了一步什么操作？请补上：10 分
    checkCudaErrors(cudaMemcpy(h_res,dev_res,n*sizeof(float),cudaMemcpyDeviceToHost));
    TOCK(exclusive_scan_and_compact);

    if (counter[0] <= n / 50) {
        printf("Result too short! %d <= %d\n", counter[0], n / 50);
        return -1;
    }
    for (int i = 0; i < counter[0]; i++) {
        if (h_res[i] < 0) {
            printf("Wrong At %d: %f < 0\n", i, h_res[i]);
            return -1;  // 突然想起了ICPC有一年队名叫“蓝翔WA掘机”的，笑不活了:)
        }
    }
    std::cout<<counter[0]<<std::endl;
    std::cout<<counter_ad[0]<<std::endl;
    printf("All Correct!\n");  // 还有个队名叫“AC自动机”的，和隔壁“WAWA大哭”对标是吧:)
    return 0;
}
