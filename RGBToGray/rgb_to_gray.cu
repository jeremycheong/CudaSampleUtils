#include "rgb_to_gray.cuh"
#include <device_launch_parameters.h>

__global__ void rgb2grayincuda(uchar3 *const d_in, unsigned char *const d_out, uint imgheight, uint imgwidth)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < imgwidth && idy < imgheight)
    {
        uchar3 rgb = d_in[idy * imgwidth + idx];
        d_out[idy * imgwidth + idx] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
    }
}

void Rgb2Gray(const cv::Mat &src, cv::Mat &gray)
{
    const uint imgheight = src.rows;
    const uint imgwidth = src.cols;

    uchar3 *d_in;
    unsigned char *d_out;

    cudaMalloc((void**)&d_in, imgwidth * imgheight * sizeof(uchar3));
    cudaMalloc((void**)&d_out, imgheight * imgwidth * sizeof(unsigned char));

    cudaMemcpy(d_in, src.data, imgwidth * imgheight * sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32); // 注意：测试机器GPU block所含的最大thread数量为1024，超过后核函数将不会被调用
    dim3 blocksPerGrid((imgwidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (imgheight + threadsPerBlock.y - 1) / threadsPerBlock.y);
    rgb2grayincuda<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, imgheight, imgwidth);
    cudaDeviceSynchronize();

    gray = cv::Mat::zeros(imgheight, imgwidth, CV_8UC1);
    cudaMemcpy(gray.data, d_out, imgwidth * imgheight * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}