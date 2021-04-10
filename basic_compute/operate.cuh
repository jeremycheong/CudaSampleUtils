#pragma once

#include <vector>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

class Operate
{
public:
    Operate()
    {
        int device_cnt = 0;
        cudaGetDeviceCount(&device_cnt);
        if (device_cnt)
        {
            gpu_mod_ = true;
            int device_id = 0;
            cudaSetDevice(device_id);
            cudaDeviceProp device_prop;
            cudaGetDeviceProperties(&device_prop, device_id);
            int max_threads_per_block = device_prop.maxThreadsPerBlock;
            int threads_per_block_dim = std::sqrt(max_threads_per_block);
            block_ = dim3(threads_per_block_dim, threads_per_block_dim);
        }
        else
            gpu_mod_ = false;

    }
    ~Operate() = default;

    // 每隔block所包含的数据完成自己的归约求和。
    Result ReduceSum(const float* data, float &result, uint32_t data_size);
    Result MatrixAdd(const float* matrix_a, float* matrix_b, float* matrix_out, uint32_t width, uint32_t height);

private:
    bool gpu_mod_;
    dim3 block_;
};
