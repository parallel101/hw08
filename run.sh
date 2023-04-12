#!/bin/bash -x
if [[ $1 = "clean" && -e build ]];then
    rm -rf build
fi
export CUDA_VISIBLE_DEVICES=1
set -e
cmake -B build
cmake --build build
build/main
