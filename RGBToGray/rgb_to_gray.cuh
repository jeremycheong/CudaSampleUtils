#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string>


__global__ void rgb2grayincuda(uchar3 * const d_in, unsigned char * const d_out, 
                                uint imgheight, uint imgwidth);

void Rgb2Gray(const cv::Mat &src, cv::Mat &gray);

void Rgb2GrayTest()
{
    std::string image_path = "../../data/lena.jpg";
    cv::Mat org_img = cv::imread(image_path);
    cv::imshow("IMAGE", org_img);
    cv::Mat gray;
    Rgb2Gray(org_img, gray);
    if (gray.empty())
        std::cout << "rgb to gray failed" << std::endl;
    cv::imshow("GRAY", gray);

    cv::waitKey(0);
    cv::destroyAllWindows();
}