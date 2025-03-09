#!/bin/bash

# Make sure the build directory exists.
BUILD_DIR=./build
if [ -d "$BUILD_DIR" ]; then
    echo "$BUILD_DIR already exists."
else
    mkdir $BUILD_DIR
fi


if [ ! -d "$BUILD_DIR" ]; then
    echo "error: $BUILD_DIR NOT exists."
    return
fi

cd $BUILD_DIR
cmake ..
make clean
make
