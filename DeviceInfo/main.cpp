#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#include "common.h"

void GetDeviceInfo()
{
    int device_cnt = 0;
    // 获取GPU数量
    cudaError_t error_id = cudaGetDeviceCount(&device_cnt);
    if (error_id != cudaSuccess)
    {
        ERROR_LOG("cudaGetDeviceCount return %d: %s", int(error_id), cudaGetErrorString(error_id));
        return;
    }

    if (!device_cnt)
    {
        WARN_LOG("There are no availabel device that support CUDA");
        return;
    }

    int dev = 0, driver_version = 0, runtime_version = 0;
    cudaSetDevice(dev);
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    INFO_LOG("Get driver version: %d.%d", driver_version / 1000, (driver_version % 100) / 10);
    INFO_LOG("Get runtime version: %d.%d", runtime_version / 1000, (runtime_version % 100) / 10);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, dev);
    INFO_LOG("Device %d: \"%s\"", dev, device_prop.name);
    INFO_LOG("CUDA Capability Major/Minor version number: %d.%d", device_prop.major, device_prop.minor);
    INFO_LOG("maxThreadsPerBlock: %d",device_prop.maxThreadsPerBlock);
    INFO_LOG("Maximun size of each dimension of a block: %d x %d x %d", device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
    INFO_LOG("Maximun size of each dimension of a grid: %d x %d x %d", device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
    INFO_LOG("multiProcessorCount: %d", device_prop.multiProcessorCount);
    INFO_LOG("maxThreadsPerMultiProcessor: %d", device_prop.maxThreadsPerMultiProcessor);
    INFO_LOG("sharedMemPerMultiprocessor: %zu", device_prop.sharedMemPerMultiprocessor);
    INFO_LOG("Maximun memory pitch %lu MB", device_prop.memPitch / int(std::pow(1024.0, 3)));
    
}




int main(int argc, char* argv[])
{
    GetDeviceInfo();

    return 0;
}