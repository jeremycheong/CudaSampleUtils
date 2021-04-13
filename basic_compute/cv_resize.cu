#include "operate.cuh"
#include <cmath>

__global__ void ResizeKernal(uchar3* src, int src_width, int src_height, uchar3* dest, int dest_width, int dest_height, float fx, float fy)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= dest_width || iy >= dest_height)
    {
        return;
    }

    int src_px = std::floor(ix / fx);
    int src_py = std::floor(iy / fy);
    src_px = src_px > src_width ? src_width : src_px;
    src_px = src_px < 0 ? 0 : src_px;
    src_py = src_py > src_height ? src_height : src_py;
    src_py = src_py < 0 ? 0 : src_py;

    dest[ix + dest_width * iy] = src[src_px + src_width * src_py];

}

Result Operate::CvResize(const cv::Mat &src, const uint32_t &dest_width, const uint32_t &dest_height, cv::Mat &dest)
{
    dest.release();

    int channel_num = src.channels();
    int width = src.cols;
    int height = src.rows;
    int step = src.step;
    printf("src image width: %d, height: %d, channels: %d, step: %d\n", width, height, channel_num, step);

    float fx = float(dest_width) / width;
    float fy = float(dest_height) / height;
    dest.create(cv::Size(dest_width, dest_height), src.type());

    dim3 grid((dest_width - 1) / block_.x + 1, (dest_height - 1) / block_.y + 1);

    uchar3* src_dev = nullptr;
    uchar3* dest_dev = nullptr;
    int src_bytes_size = height * width * sizeof(uchar3);
    int dest_bytes_size = dest.rows * dest.cols * sizeof(uchar3);
    cudaMalloc((void**)&src_dev, src_bytes_size);
    cudaMalloc((void**)&dest_dev, dest_bytes_size);

    cudaMemcpy(src_dev, src.data, src_bytes_size, cudaMemcpyHostToDevice);

    ResizeKernal<<<grid, block_>>>(src_dev, width, height, dest_dev, dest_width, dest_height, fx, fy);
    cudaDeviceSynchronize();

    cudaMemcpy(dest.data, dest_dev, dest_bytes_size, cudaMemcpyDeviceToHost);

    cudaFree(src_dev);
    cudaFree(dest_dev);

    return SUCCESS;
}