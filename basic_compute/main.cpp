#include "common.h"
#include "operate.cuh"

#include <iostream>

class Interface
{
public:
    Interface() = default;
    virtual ~Interface() {INFO_LOG("Interface destroy!"); };
    virtual void PreProcess() = 0;
    virtual void Infer() = 0;
    virtual void PostProcess() = 0;
};

class ModelInterface : public Interface
{
public:
    ~ModelInterface() {
        INFO_LOG("ModelInterface destroy!");
    }
    void PreProcess() final {
        INFO_LOG("ModelInterface PreProcess");
    }
    void Infer() final {
        INFO_LOG("ModelInterface Infer");
    }
    void PostProcess() final {
        INFO_LOG("ModelInterface PostProcess");
    }
};

void TestInterface()
{
    Interface* model_interface = nullptr;
    model_interface = new ModelInterface();
    model_interface->PreProcess();
    model_interface->Infer();
    model_interface->PostProcess();
    if (model_interface)
    {
        delete(model_interface);
        model_interface = nullptr;
    }
    
    INFO_LOG("TestInterface Done");
}

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

int main(int argc, char* argv[])
{
    TestReduceSum(argc, argv);
    // TestMatrixAdd();
    // TestInterface();

    return 0;
}