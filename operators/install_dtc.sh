#!/bin/bash

# Install DTC-SpMM

export DTC_HOME=$(pwd)/DTC-SpMM

# Build Sputnik (dependency)
echo "Building glog..."
cd DTC-SpMM/third_party/glog
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=${DTC_HOME}/third_party/glog/build ..
make -j
make install

export GLOG_PATH=${DTC_HOME}/third_party/glog
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GLOG_PATH/build/lib
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$GLOG_PATH/build/include
export LIBRARY_PATH=$LD_LIBRARY_PATH:$GLOG_PATH/build/lib

echo "Building Sputnik..."
cd ${DTC_HOME}/third_party/sputnik
mkdir -p build && cd build
cmake .. -DGLOG_INCLUDE_DIR=$GLOG_PATH/build/include -DGLOG_LIBRARY=$GLOG_PATH/build/lib/libglog.so -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF -DCUDA_ARCHS="89;86"
make -j

export SPUTNIK_PATH=${DTC_HOME}/third_party/sputnik
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SPUTNIK_PATH/build/sputnik

# Build DTC-SpMM
echo "Building DTC-SpMM..."
cd ${DTC_HOME}/DTC-SpMM
TORCH_CUDA_ARCH_LIST="8.6 8.9" python setup.py install

cd ../..
echo "DTC-SpMM installation complete."
