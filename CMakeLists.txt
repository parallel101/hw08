cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# 如果需要指定显卡版本号的话：
# I am RTX 2080
set(CMAKE_CUDA_ARCHITECTURES 75)

project(hellocmake LANGUAGES CXX CUDA)

add_executable(main main.cu)
#解决分离 __device__ 函数的声明和定义
set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)


# 为了只对 .cu 文件开启这个开关，可以用 CMake 的生成器表达式
# 限制 flag 只对 CUDA 源码生效，这样可以混合其他 .cpp 文件也不会发生 gcc 报错的情况了
target_include_directories(main PUBLIC include)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)