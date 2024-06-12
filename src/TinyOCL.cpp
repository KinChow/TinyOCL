/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-13 01:45:39 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-13 03:04:07
 */

#include <iostream>
#include <vector>
#include <CL/cl.h>
#include "TinyOCL.h"

#define CHECK_OPENCL_ERROR(ret, msg) \
    do { \
        if (ret != CL_SUCCESS) { \
            std::cout << "OpenCL error: " << msg << " (" << ret << ")" << std::endl; \
            return false; \
        } \
    } while (0)

class TinyOCL::TinyOCLImpl {
public:
    TinyOCLImpl();
    ~TinyOCLImpl() = default;
    TinyOCLImpl(const TinyOCLImpl &) = delete;
    TinyOCLImpl &operator=(const TinyOCLImpl &) = delete;

private:
    bool Init();

    std::vector<cl_device_id> devices_;
    std::unique_ptr<_cl_context, decltype(&clReleaseContext)> context_{nullptr, clReleaseContext};
    std::unique_ptr<_cl_command_queue, decltype(&clReleaseCommandQueue)> command_queue_{nullptr, clReleaseCommandQueue};
};

TinyOCL::TinyOCLImpl::TinyOCLImpl() {
    if (!Init()) {
        std::cout << "Failed to initialize TinyOCL" << std::endl;
    }
}

bool TinyOCL::TinyOCLImpl::Init() {
    cl_uint num_platforms;
    cl_int ret;
    ret = clGetPlatformIDs(0, nullptr, &num_platforms);
    CHECK_OPENCL_ERROR(ret, "Failed to get number of platforms");
    if (num_platforms == 0) {
        std::cout << "No OpenCL platforms found" << std::endl;
        return false;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    ret = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    CHECK_OPENCL_ERROR(ret, "Failed to get platform IDs");

    for (const auto &platform : platforms) {
        cl_uint num_devices;
        ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (num_devices == 0) {
            continue;
        }
        devices_.resize(num_devices);
        ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices_.data(), nullptr);
        CHECK_OPENCL_ERROR(ret, "Failed to get device IDs");
        std::cout << "GPU device found" << std::endl;
        context_.reset(clCreateContext(nullptr, num_devices, devices_.data(), nullptr, nullptr, &ret));
        CHECK_OPENCL_ERROR(ret, "Failed to create context");

        cl_command_queue_properties properties[] = {0};
        command_queue_.reset(clCreateCommandQueueWithProperties(context_.get(), devices_[0], properties, &ret));
        CHECK_OPENCL_ERROR(ret, "Failed to create command queue");
        
        return true;
    }
    std::cout << "No GPU devices found" << std::endl;
    return false;
}

TinyOCL &TinyOCL::GetInstance() {
    static TinyOCL instance;
    return instance;
}

TinyOCL::TinyOCL() : impl_(std::make_unique<TinyOCLImpl>()) {}
