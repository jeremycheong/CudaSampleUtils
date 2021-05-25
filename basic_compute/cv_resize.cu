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

//待求点的像素值 = 左上点的像素值 x 右下矩形的面积 + 左下点的像素值 x 右上矩形的面积 + 右上点的像素值 x 左下矩形的面积 + 右下点的像素值 x 左上矩形的面积。
__global__ void BiLinearKernal(uchar3* src, int src_width, int src_height, uchar3* dest, int dest_width, int dest_height)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= dest_width || iy >= dest_height)
    {
        return;
    }

    float scale_x = (float)(src_width) / dest_width;
    float scale_y = (float)(src_height) / dest_height;

    float px = (ix + 0.5) * scale_x - 0.5;
    float py = (iy + 0.5) * scale_y - 0.5;

    int src_p00_x = std::floor(px);
    if (src_p00_x < 0)
    {
        src_p00_x = 0;
        px = 0;
    }
    if (src_p00_x >= src_width - 1)
    {
        src_p00_x = src_width - 2;
        px = src_p00_x;
    }

    float dist_p_p00_x = px - src_p00_x;
    float dist_p_p01_x = 1.f - dist_p_p00_x;

    int src_p00_y = std::floor(py); 
    if (src_p00_y < 0)
    {
        src_p00_y = 0;
        py = 0;
    }
    if (src_p00_y >= src_height - 1)
    {
        src_p00_y = src_height - 2;
        py = src_p00_y;
    }

    float dist_p_p00_y = py - src_p00_y;
    float dist_p_p10_y = 1.f - dist_p_p00_y;

    float alph00 = dist_p_p00_x * dist_p_p00_y; // 左上面积
    float alph01 = dist_p_p01_x * dist_p_p00_y; // 右上面积
    float alph10 = dist_p_p00_x * dist_p_p10_y; // 左下面积
    float alph11 = dist_p_p01_x * dist_p_p10_y; // 右下面积

    dest[ix + dest_width * iy].x = src[src_p00_x + src_width * src_p00_y].x * alph11
                                    + src[src_p00_x + 1 + src_width * src_p00_y].x * alph10
                                    + src[src_p00_x + src_width * (src_p00_y + 1)].x * alph01
                                    + src[src_p00_x + 1 + src_width * (src_p00_y + 1)].x * alph00;

    dest[ix + dest_width * iy].y = src[src_p00_x + src_width * src_p00_y].y * alph11
                                    + src[src_p00_x + 1 + src_width * src_p00_y].y * alph10
                                    + src[src_p00_x + src_width * (src_p00_y + 1)].y * alph01
                                    + src[src_p00_x + 1 + src_width * (src_p00_y + 1)].y * alph00;

    dest[ix + dest_width * iy].z = src[src_p00_x + src_width * src_p00_y].z * alph11
                                    + src[src_p00_x + 1 + src_width * src_p00_y].z * alph10
                                    + src[src_p00_x + src_width * (src_p00_y + 1)].z * alph01
                                    + src[src_p00_x + 1 + src_width * (src_p00_y + 1)].z * alph00;
}

Result Operate::CvBiLinearResize(const cv::Mat &src, const uint32_t &dest_width, const uint32_t &dest_height, cv::Mat &dest)
{
    dest.release();

    int channel_num = src.channels();
    int width = src.cols;
    int height = src.rows;
    int step = src.step;
    printf("src image width: %d, height: %d, channels: %d, step: %d\n", width, height, channel_num, step);

    dest.create(cv::Size(dest_width, dest_height), src.type());

    dim3 grid((dest_width - 1) / block_.x + 1, (dest_height - 1) / block_.y + 1);

    uchar3* src_dev = nullptr;
    uchar3* dest_dev = nullptr;
    int src_bytes_size = height * width * sizeof(uchar3);
    int dest_bytes_size = dest.rows * dest.cols * sizeof(uchar3);
    cudaMalloc((void**)&src_dev, src_bytes_size);
    cudaMalloc((void**)&dest_dev, dest_bytes_size);

    cudaMemcpy(src_dev, src.data, src_bytes_size, cudaMemcpyHostToDevice);

    BiLinearKernal<<<grid, block_>>>(src_dev, width, height, dest_dev, dest_width, dest_height);
    cudaDeviceSynchronize();

    cudaMemcpy(dest.data, dest_dev, dest_bytes_size, cudaMemcpyDeviceToHost);

    cudaFree(src_dev);
    cudaFree(dest_dev);

    return SUCCESS;
}

__global__ void PadBiLinearKernal(uchar3* src, int src_width, int src_height, uchar3* dest, int dest_width, int dest_height, int roi_width, int roi_height)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= roi_width || iy >= roi_height)
    {
        return;
    }

    float scale_x = (float)(src_width) / roi_width;
    float scale_y = (float)(src_height) / roi_height;

    float px = (ix + 0.5) * scale_x - 0.5;
    float py = (iy + 0.5) * scale_y - 0.5;

    int src_p00_x = std::floor(px);
    if (src_p00_x < 0)
    {
        src_p00_x = 0;
        px = 0;
    }
    if (src_p00_x >= src_width - 1)
    {
        src_p00_x = src_width - 2;
        px = src_p00_x;
    }

    float dist_p_p00_x = px - src_p00_x;
    float dist_p_p01_x = 1.f - dist_p_p00_x;

    int src_p00_y = std::floor(py); 
    if (src_p00_y < 0)
    {
        src_p00_y = 0;
        py = 0;
    }
    if (src_p00_y >= src_height - 1)
    {
        src_p00_y = src_height - 2;
        py = src_p00_y;
    }

    float dist_p_p00_y = py - src_p00_y;
    float dist_p_p10_y = 1.f - dist_p_p00_y;

    float alph00 = dist_p_p00_x * dist_p_p00_y; // 左上面积
    float alph01 = dist_p_p01_x * dist_p_p00_y; // 右上面积
    float alph10 = dist_p_p00_x * dist_p_p10_y; // 左下面积
    float alph11 = dist_p_p01_x * dist_p_p10_y; // 右下面积

    dest[ix + dest_width * iy].x = src[src_p00_x + src_width * src_p00_y].x * alph11
                                    + src[src_p00_x + 1 + src_width * src_p00_y].x * alph10
                                    + src[src_p00_x + src_width * (src_p00_y + 1)].x * alph01
                                    + src[src_p00_x + 1 + src_width * (src_p00_y + 1)].x * alph00;

    dest[ix + dest_width * iy].y = src[src_p00_x + src_width * src_p00_y].y * alph11
                                    + src[src_p00_x + 1 + src_width * src_p00_y].y * alph10
                                    + src[src_p00_x + src_width * (src_p00_y + 1)].y * alph01
                                    + src[src_p00_x + 1 + src_width * (src_p00_y + 1)].y * alph00;

    dest[ix + dest_width * iy].z = src[src_p00_x + src_width * src_p00_y].z * alph11
                                    + src[src_p00_x + 1 + src_width * src_p00_y].z * alph10
                                    + src[src_p00_x + src_width * (src_p00_y + 1)].z * alph01
                                    + src[src_p00_x + 1 + src_width * (src_p00_y + 1)].z * alph00;
}

Result Operate::CvPadResize(const cv::Mat &src, const uint32_t &dest_width, const uint32_t &dest_height, 
                                    float &scale, cv::Rect &paste_roi, cv::Mat &dest)
{
    dest.release();

    int channel_num = src.channels();
    int width = src.cols;
    int height = src.rows;
    int step = src.step;
    printf("src image width: %d, height: %d, channels: %d, step: %d\n", width, height, channel_num, step);

    dest.create(cv::Size(dest_width, dest_height), src.type());
    cv::Vec3b fill_val(127, 127, 127);
    dest.setTo(fill_val);

    PadResize(width, height, dest_width, dest_height, paste_roi, scale);

    dim3 grid((dest_width - 1) / block_.x + 1, (dest_height - 1) / block_.y + 1);

    uchar3* src_dev = nullptr;
    uchar3* dest_dev = nullptr;
    int src_bytes_size = height * width * sizeof(uchar3);
    int dest_bytes_size = dest.rows * dest.cols * sizeof(uchar3);
    cudaMalloc((void**)&src_dev, src_bytes_size);
    cudaMalloc((void**)&dest_dev, dest_bytes_size);

    cudaMemcpy(src_dev, src.data, src_bytes_size, cudaMemcpyHostToDevice);

    PadBiLinearKernal<<<grid, block_>>>(src_dev, width, height, dest_dev, dest_width, dest_height, paste_roi.width, paste_roi.height);
    cudaDeviceSynchronize();

    cudaMemcpy(dest.data, dest_dev, dest_bytes_size, cudaMemcpyDeviceToHost);

    cudaFree(src_dev);
    cudaFree(dest_dev);

    return SUCCESS;

}

Result Operate::PadResize(const uint32_t &image_width, const uint32_t &image_height, const uint32_t &dest_width, const uint32_t &dest_height, 
                            cv::Rect &dest_roi, float &scale)
{
    scale = 1.0f;
    if (image_width <= (int)dest_width && image_height <= (int)dest_height)
    {
        dest_roi.x = 0;
        dest_roi.y = 0;
        dest_roi.width = image_width;
        dest_roi.height = image_height;
    }
    else if (float(image_width) / image_height >= float(dest_width) / dest_height)  // 按照width缩放
    {
        scale = float(image_width) / dest_width;
        dest_roi.x = 0;
        dest_roi.y = 0;
        dest_roi.width = dest_width;
        dest_roi.height = std::round(image_height / scale);
    }
    else    // 按照height缩放
    {
        scale = float(image_height) / dest_height;
        dest_roi.x = 0;
        dest_roi.y = 0;
        dest_roi.width = std::round(image_width / scale);
        dest_roi.height = dest_height;
    }

    return SUCCESS;
}