#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"
#include "opencv2/opencv.hpp"

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

    // 每个block所包含的数据完成自己的归约求和。
    Result ReduceSum(const float* data, float &result, uint32_t data_size);
    Result MatrixAdd(const float* matrix_a, float* matrix_b, float* matrix_out, uint32_t width, uint32_t height);

    // matrix_T_data 需外部初始化为全0矩阵
    Result SparseMatrixTranspose(const float* matrix_data, float* matrix_T_data, uint32_t width, uint32_t height);

    // 最近邻插值实现resize
    Result CvResize(const cv::Mat &src, const uint32_t &dest_width, const uint32_t &dest_height, cv::Mat &dest);

    // 双线性插值实现resize
    Result CvBiLinearResize(const cv::Mat &src, const uint32_t &dest_width, const uint32_t &dest_height, cv::Mat &dest);

    // 等比例resize和减均值除方差
    Result CvPadResize(const cv::Mat &src, const uint32_t &dest_width, const uint32_t &dest_height, 
                                    float &scale, cv::Rect &paste_roi, cv::Mat &dest);

private:
    Result PadResize(const uint32_t &image_width, const uint32_t &image_height, const uint32_t &dest_width, const uint32_t &dest_height, 
                            cv::Rect &dest_roi, float &scale);

private:
    bool gpu_mod_;
    dim3 block_;
};

