#pragma once
#include <cuda_runtime.h>
#include <random>

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)


// static void HandleError(cudaError_t err,
//                         const char *file,
//                         int line)
//                         {
//                             if(err != cudaSuccess)
//                             {
//                                 printf("%s in %s:%d\n",
//                                 cudaGetErrorString(err),
//                                 file, line);
//                                 exit(EXIT_FAILURE);
//                             }
//                         }

// #define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))


enum Result
{
    SUCCESS,
    FAILED
};

class Common
{
public:
    Common() = delete;
    ~Common() = default;
    static void GenerateRangeData(float* data, uint32_t data_size, float alph = 0.f);
    static void GenerateData(unsigned char *image_data, uint32_t data_size); 
    static void GenerateData(float *data, uint32_t data_size); 
};
