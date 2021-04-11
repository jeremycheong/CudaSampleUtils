#include "operate.cuh"

void __global__ SparseTransposeKernal(float* matrix, float* matrix_T_data, uint32_t width, uint32_t height)
{
    // 映射到二维矩阵中的实际坐标
    uint32_t ix = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t iy = threadIdx.y + blockIdx.y * blockDim.y;
    // 检查合法性
    if (ix >= width || iy >= height)
        return;
    
    // 映射到一维数组中的索引
    uint32_t g_tid = ix + iy * (width);

    // transpose 后二维矩阵的坐标以及其对应的一维数组中的索引
    uint32_t ox = iy;
    uint32_t oy = ix;
    uint32_t g_out_tid = ox + oy * (height);

    // 将非零数值进行交换
    if (matrix[g_tid] != 0)
        matrix_T_data[g_out_tid] = matrix[g_tid];
}

Result Operate::SparseMatrixTranspose(const float* matrix_data, float* matrix_T_data, uint32_t width, uint32_t height)
{
    dim3 grid((width - 1) / block_.x + 1, (height - 1) / block_.y + 1);

    float* dev_matrix = nullptr;
    float* dev_matrix_T = nullptr;
    uint32_t data_bytes_size = width * height * sizeof(float);
    cudaMalloc((void**)&dev_matrix, data_bytes_size);
    cudaMalloc((void**)&dev_matrix_T, data_bytes_size);

    cudaMemcpy(dev_matrix, matrix_data, data_bytes_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix_T, matrix_T_data, data_bytes_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    SparseTransposeKernal<<<grid, block_>>>(dev_matrix, dev_matrix_T, width, height);

    cudaMemcpy(matrix_T_data, dev_matrix_T, data_bytes_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(dev_matrix);
    cudaFree(dev_matrix_T);

    return SUCCESS;
}