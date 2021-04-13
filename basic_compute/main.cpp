#include "common.h"
#include "operate.cuh"

#include <iostream>
#include <cstring>

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
    Operate op;
    op.CvResize(input_image, input_image.cols * 2, input_image.rows * 2, resized);

    cv::imshow("lena", input_image);
    cv::imshow("resized", resized);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char* argv[])
{
    // TestReduceSum(argc, argv);
    // TestMatrixAdd();
    // TestSparseMatrixTranspose();
    TestCvResize();

    return 0;
}