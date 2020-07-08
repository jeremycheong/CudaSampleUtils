#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string>


void rgb2graycpu(uchar3* const in, unsigned char* out, uint imgheight, uint imgwidth)
{
    for (uint i = 0; i < imgheight; i ++)
    {
        for (uint j = 0; j < imgwidth; j ++)
        {
            uchar3 in_data = in[i * imgwidth + j];
            out[i * imgwidth + j] = 0.299f * in_data.x + 0.587f * in_data.y + 0.114f * in_data.z;
        }
    }
}

void Rgb2Gray(const cv::Mat &src, cv::Mat &gray);

void Rgb2GrayCpu(const cv::Mat &src, cv::Mat &gray)
{
    const uint imgheight = src.rows;
    const uint imgwidth = src.cols;
    gray = cv::Mat::zeros(imgheight, imgwidth, CV_8UC1);
    rgb2graycpu((uchar3*)src.data, gray.data, imgheight, imgwidth);
}

void Rgb2GrayTest()
{
    std::string image_path = "../../data/lena.jpg";
    cv::Mat org_img = cv::imread(image_path);
    cv::imshow("IMAGE", org_img);
    cv::Mat gray;
    Rgb2Gray(org_img, gray);
    // Rgb2GrayCpu(org_img, gray);
    if (gray.empty())
        std::cout << "rgb to gray failed" << std::endl;
    cv::imshow("GRAY", gray);

    cv::waitKey(0);
    cv::destroyAllWindows();
}