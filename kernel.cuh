#define THREADS_PER_BLOCK 512

template<class T,int thread_per_block=THREADS_PER_BLOCK>
void scan(T *d_in,T *d_out,int length);

template<class T,int thread_per_block=THREADS_PER_BLOCK>
__global__ void singleBlockScan(T *d_in,T *d_out,int length);

template<class T,int thread_per_block=THREADS_PER_BLOCK>
void largeDataScan(T *d_in,T *d_out,int length);

template<class T,int thread_per_block=THREADS_PER_BLOCK>
__global__ void preScan(T *d_in,T *d_out,T *block_offset);

template<class T,int thread_per_block=THREADS_PER_BLOCK>
__global__ void postScan(T *d_out,T *block_offset);

template<class T>
__global__ void procLeft(T *d_out,T *num1,T *num2, int length);

template<class T>
__global__ void filter(T *arr,int *isPos,int n);

template<class T>
__global__ void Compact(T *arr,T *res,int *isPos,int *index,int n);