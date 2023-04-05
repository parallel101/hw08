#!/bin/bash -x
if [[ $1 = "clean" && -e build ]];then
    rm -rf build
fi
set -e
cmake -B build
cmake --build build
build/main
