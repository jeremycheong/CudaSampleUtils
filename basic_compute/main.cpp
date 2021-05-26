#include "common.h"
#include "operate.cuh"

#include <iostream>
#include <cstring>
#include <memory>

void TestMatrixAdd()
{
    uint32_t width = 1 << 6;
    uint32_t height = 1 << 5;

    float* matrix_a = new float[width * height];
    float* matrix_b = new float[width * height];

    Common::GenerateRangeData(matrix_a, width * height);
    Common::GenerateRangeData(matrix_b, width * height, 2);
    INFO_LOG("Generate data success");
    float* matrix_out = new float[width * height];
    Operate op;
    op.MatrixAdd(matrix_a, matrix_b, matrix_out, width, height);

    for (uint32_t i = 0; i < height; ++ i)
    {
        float* data_ptr = matrix_out + i * width;
        std::cout << "[ ";
        for (uint32_t j = 0; j < width; ++ j)
        {
            std::cout << data_ptr[j] << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "========================================================" << std::endl;
    }

    delete[](matrix_a);
    delete[](matrix_b);
    delete[](matrix_out);
    INFO_LOG("Done");
}

int TestReduceSum(int argc, char* argv[])
{
    uint32_t data_size = 1 << 10;
    INFO_LOG("data_size: %u", data_size);

    float* data = new float[data_size];
    Common::GenerateRangeData(data, data_size);
    INFO_LOG("Generate data success");
    float sum = 0.f;
    for (int i = 0; i < data_size; i ++)
    {
        sum += data[i];
    }
    INFO_LOG("cpu sum: %f", sum);

    sum = 0.f;
    Operate op;
    op.ReduceSum(data, sum, data_size);
    INFO_LOG("reduce sum: %f", sum);

    delete[](data);
    INFO_LOG("Done");
    return 0;
}

void TestSparseMatrixTranspose()
{
    uint32_t data_size = 845;
    INFO_LOG("data_size: %u", data_size);

    uint32_t width = 65;
    uint32_t height = data_size / width;

    float* data = new float[data_size];
    Common::GenerateRangeData(data, data_size);
    INFO_LOG("Generate data success");
    float* out_data = new float[data_size];
    std::memset(out_data, 0, data_size * sizeof(float));

    Operate op;
    op.SparseMatrixTranspose(data, out_data, width, height);
    uint32_t out_width = height;
    uint32_t out_height = width;

    for (uint32_t i = 0; i < out_height; i ++)
    {
        std::cout << "========================================== " << i << std::endl;
        float* row_data = out_data + i * out_width;
        std::cout << "[ ";
        for (int j = 0; j < out_width; j ++)
        {
            std::cout << row_data[j] << ", ";
        }
        std::cout << "]" << std::endl;
    }
    delete[](data);
    delete[](out_data);

    INFO_LOG("Done");
}

void TestCvResize()
{
    cv::Mat input_image = cv::imread("../data/lena.jpg");
    cv::Mat resized;
    cv::Mat cv_resized;
    // cv::resize(input_image, cv_resized, cv::Size(input_image.cols * 2, input_image.rows * 2), 0.f, 0.f, cv::INTER_NEAREST);
    cv::resize(input_image, cv_resized, cv::Size(640, 640), 0.f, 0.f, cv::INTER_LINEAR);
    cv::imshow("cv resize", cv_resized);

    Operate op;
    // op.CvResize(input_image, input_image.cols * 2, input_image.rows * 2, resized);
    // op.CvBiLinearResize(input_image, input_image.cols * 2, input_image.rows * 2, resized);
    float scale;
    cv::Rect paste_roi;
    op.CvPadResize(cv_resized, 640, 384, scale, paste_roi, resized);

    cv::imwrite("./pad_resized.jpg", resized);

    // cv::imshow("lena", input_image);
    // cv::imshow("resized", resized);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void TestCvPadResizeNormal()
{
    cv::Mat input_image = cv::imread("../data/lena.jpg");

    int width = input_image.cols;
    int height = input_image.rows;
    int c = input_image.channels();
    int resized_out_w = 640;
    int resized_out_h = 384;
    cv::Mat resized;

    Operate op;
    // opencv 方式进行预处理
    float scale;
    cv::Rect paste_roi;
    op.CvPadResize(input_image, resized_out_w, resized_out_h, scale, paste_roi, resized);
    // cv::imshow("Resized", resized);
    
    cv::Mat input_float;
    resized.convertTo(input_float, CV_32FC3, 1.0f / 255);
    float *input_data = new float[resized.cols * resized.rows * resized.channels()];
    cv::Mat channel_r(cv::Size(resized_out_w, resized_out_h), CV_32FC1, input_data);
    cv::Mat channel_g(cv::Size(resized_out_w, resized_out_h), CV_32FC1, input_data + resized_out_w * resized_out_h);
    cv::Mat channel_b(cv::Size(resized_out_w, resized_out_h), CV_32FC1, input_data + resized_out_w * resized_out_h * 2);
    std::vector<cv::Mat> channels = {channel_b, channel_g, channel_r};
    cv::split(input_float, channels);       // bgr
    
    // cuda 方式进行预处理
    std::cout << "=================================" << std::endl;
    uchar3* input_dev = nullptr;
    float* output_dev = nullptr;
    int input_bytes_size = height * width * sizeof(uchar3);
    int output_bytes_size = resized_out_w * resized_out_h * c * sizeof(float);
    cudaMalloc((void**)&input_dev, input_bytes_size);
    cudaMalloc((void**)&output_dev, output_bytes_size);

    // cv::imshow("cuda_input_image", input_image);
    cudaMemcpy(input_dev, input_image.data, input_bytes_size, cudaMemcpyHostToDevice);

    std::vector<int> means = {0, 0, 0};
    std::vector<float> vars = {1.0f / 255, 1.0f / 255, 1.0f / 255};
    float scale_g;
    op.CvPadResizeGpu(input_dev, width, height, resized_out_w, resized_out_h, means, vars, scale_g, output_dev);

    float* input_data_g = new float[resized_out_w * resized_out_h * c];
    cudaMemcpy(input_data_g, output_dev, output_bytes_size, cudaMemcpyDeviceToHost);
    cudaFree(input_dev);
    cudaFree(output_dev);

    // 判断是否一致
    int data_size = resized_out_w * resized_out_h * c;
    INFO_LOG("=====================");
    for (size_t i = 0; i < data_size; i ++)
    {
        // std::cout << input_data[i] << std::endl;
        // if (input_data_g[i])
        // {
        //     std::cout << input_data_g[i] << std::endl;
        // }

        if (input_data[i] != input_data_g[i])
        {
            INFO_LOG("opencv process [%zu] result is not equal with gpu process result [%0.4f != %0.4f]", i, input_data[i], input_data_g[i]);
        }
    }

    delete[](input_data);
    delete[](input_data_g);

    // cv::waitKey(0);
}

int main(int argc, char* argv[])
{
    // TestReduceSum(argc, argv);
    // TestMatrixAdd();
    // TestSparseMatrixTranspose();
    // TestCvResize();
    TestCvPadResizeNormal();

    return 0;
}