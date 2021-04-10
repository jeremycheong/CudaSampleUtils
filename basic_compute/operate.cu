#include "operate.cuh"

void __global__ ReduceSumKernal(float* data, float* out, uint32_t data_size)
{
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int g_tid = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (g_tid >= data_size)
        return;
    
    float* block_data_ptr = data + block_id * (blockDim.x * blockDim.y);

    int tid_in_block = threadIdx.x + threadIdx.y * blockDim.x;

    for (int stride = 1; stride < blockDim.x * blockDim.y; stride *= 2)
    {
        if (tid_in_block % (2 * stride) == 0)
        {
            block_data_ptr[tid_in_block] += block_data_ptr[tid_in_block + stride];
        }
        __syncthreads();
    }

    if (tid_in_block == 0)
    {
        out[block_id] = block_data_ptr[0];
    }


}

Result Operate::ReduceSum(const float* data, float &result, uint32_t data_size)
{
    dim3 grid((data_size - 1) / (block_.x * block_.y) + 1, 1);

    uint32_t out_size = grid.x * grid.y; 
    float* out = new float[out_size * sizeof(float)];

    float* dev_data = nullptr;
    float* dev_out = nullptr;
    uint32_t bytes_size = data_size * sizeof(float);
    cudaMalloc((void**)&dev_data, bytes_size);
    cudaMalloc((void**)&dev_out, out_size * sizeof(float));

    

    cudaMemcpy(dev_data, data, bytes_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    ReduceSumKernal<<<grid, block_>>>(dev_data, dev_out, data_size);
    cudaMemcpy(out, dev_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    result = 0.f;
    for (uint32_t i = 0; i < out_size; i ++)
    {
        result += out[i];
    }
    delete[](out);
    
    cudaFree(dev_data);
    cudaFree(dev_out);

    return SUCCESS;
}

void __global__ MatrixAddKernal(float* dev_a, float* dev_b, float* dev_out, uint32_t width, uint32_t height)
{
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int thread_id = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (thread_id < width * height)
    {
        dev_out[thread_id] = dev_a[thread_id] + dev_b[thread_id];
    }
}

Result Operate::MatrixAdd(const float* matrix_a, float* matrix_b, float* matrix_out, uint32_t width, uint32_t height)
{
    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_out = nullptr;
    uint32_t bytes_size = width * height * sizeof(float);

    cudaMalloc((void**)&dev_a, bytes_size);
    cudaMalloc((void**)&dev_b, bytes_size);
    cudaMalloc((void**)&dev_out, bytes_size);

    cudaMemcpy(dev_a, matrix_a, bytes_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, matrix_b, bytes_size, cudaMemcpyHostToDevice);

    dim3 grid((width - 1) / block_.x + 1, (height - 1) / block_.y + 1);

    MatrixAddKernal<<<grid, block_>>>(dev_a, dev_b, dev_out, width, height);

    cudaMemcpy(matrix_out, dev_out, bytes_size, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_out);

    return SUCCESS;
}