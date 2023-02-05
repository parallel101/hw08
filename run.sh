#!/bin/sh
set -e
cmake -B build
cmake --build build
# build/main
build/Debug/main.exe
