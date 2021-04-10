#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 情况1：grid划分成1维，block划分为1维。
__device__ int GetGlobalIdx_1D_1D()
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    return threadId;
}

// 情况2：grid划分成1维，block划分为2维。
__device__ int GetGlobalIdx_1D_2D()
{
    int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    return threadId;
}

// 情况3：grid划分成1维，block划分为3维。
__device__ int GetGlobalIdx_1D_3D()
{
    int threadId = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    return threadId;
}

// 情况4：grid划分成2维，block划分为1维。
__device__ int GetGlobalIdx_2D_1D()
{
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}

// 情况5：grid划分成2维，block划分为2维。
__device__ int GetGlobalIdx_2D_2D()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

// 情况6：grid划分成2维，block划分为3维。
__device__ int GetGlobalIdx_2D_3D()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

// 情况7：grid划分成3维，block划分为1维。
__device__ int GetGlobalIdx_3D_1D()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}

// 情况8：grid划分成3维，block划分为2维。
__device__ int GetGlobalIdx_3D_2D()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

// 情况9：grid划分成3维，block划分为3维。
__device__ int GetGlobalIdx_3D_3D()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}