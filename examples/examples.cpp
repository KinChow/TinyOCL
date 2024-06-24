/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-12 23:18:00 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-24 23:25:42
 */

#include <iostream>
#include "TinyOCL.h"

int main()
{
    auto kernel = TinyOCL::Executor::GetInstance().CreateKernel("cl/calc.cl", "add", {});
    auto buffer0 = TinyOCL::Executor::GetInstance().CreateBuffer(10 * sizeof(float));
    auto buffer1 = TinyOCL::Executor::GetInstance().CreateBuffer(10 * sizeof(float));
    auto buffer2 = TinyOCL::Executor::GetInstance().CreateBuffer(10 * sizeof(float));
    auto data0 = buffer0->GetHostPtr<float *>();
    auto data1 = buffer1->GetHostPtr<float *>();
    auto data2 = buffer2->GetHostPtr<float *>();
    if (!data0 || !data1 || !data2) {
        std::cout << "Failed to get host pointer" << std::endl;
        return -1;
    }
    for (int i = 0; i < 10; i++) {
        data0[i] = i;
        data1[i] = i + 1;
        data2[i] = 0;
    }
    bool ret = kernel->Run({10}, {10}, false, buffer0->GetClMem(), buffer1->GetClMem(), buffer2->GetClMem());
    if (!ret) {
        std::cout << "Failed to run kernel" << std::endl;
    }
    for (int i = 0; i < 10; i++) {
        std::cout << data2[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}