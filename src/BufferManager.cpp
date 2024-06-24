/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-17 02:55:21 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-17 03:52:20
 */

#include <iostream>
#include "utils.h"
#include "BufferManager.h"

namespace TinyOCL {

BufferManager::BufferManager(cl_context context, cl_command_queue queue) : context_(context), queue_(queue) {}

BufferManager::~BufferManager()
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "Release " << buffers_.size() << " buffers" << std::endl;
    for (auto buffer : buffers_) {
        clReleaseMemObject(buffer);
        std::cout << "Release buffer " << buffer << std::endl;
    }
}

cl_mem BufferManager::Create(size_t size)
{
    std::lock_guard<std::mutex> lock(mutex_);
    cl_int ret;
    cl_mem buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, nullptr, &ret);
    CHECK_OPENCL_ERROR_RETURN_NULL(ret, "Failed to create buffer");
    buffers_.emplace(buffer);
    return buffer;
}

void BufferManager::Release(cl_mem buffer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto buffers_iter = buffers_.find(buffer);
    if (buffers_iter == buffers_.end()) {
        return;
    }
    buffers_.erase(buffer);
    clReleaseMemObject(buffer);
}

}  // namespace TinyOCL