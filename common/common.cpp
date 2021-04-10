#include "common.h"

void Common::GenerateRangeData(float* data, uint32_t data_size, float alph)
{
    for (uint32_t i = 0; i < data_size; ++ i)
    {
        data[i] = (float)i + alph;
    }

}

void Common::GenerateData(unsigned char *image_data, uint32_t data_size)
{
    std::random_device rd;
    std::mt19937 mt(rd());  // 梅森旋转算法
    std::uniform_int_distribution<> dis(0, 255);
    for(size_t i = 0; i < data_size; ++i)
    {
        image_data[i] = (unsigned char)dis(mt);
    }
}

void Common::GenerateData(float *data, uint32_t data_size)
{
    std::random_device rd;
    std::mt19937 mt(rd());  // 梅森旋转算法
    std::uniform_real_distribution<float> dis(-1, 1);
    for(size_t i = 0; i < data_size; ++i)
    {
        data[i] = dis(mt);
    }
}