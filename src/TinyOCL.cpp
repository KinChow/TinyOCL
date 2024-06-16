/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-13 01:45:39 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-17 03:50:28
 */

#include <iostream>
#include <vector>
#include <CL/cl.h>
#include "utils.h"
#include "BufferManager.h"
#include "ProgramManager.h"
#include "TinyOCL.h"

namespace tinyocl {
class Kernel::KernelImpl final {
public:
    explicit KernelImpl(cl_command_queue queue, cl_kernel kernel);
    ~KernelImpl() = default;
    KernelImpl() = delete;
    KernelImpl(const KernelImpl &) = delete;
    KernelImpl &operator=(const KernelImpl &) = delete;
    KernelImpl(KernelImpl &&) = delete;
    KernelImpl &operator=(KernelImpl &&) = delete;

    bool SetArg(cl_uint index, size_t size, const void *value) const;
    bool Run(const std::vector<size_t> &global_size, const std::vector<size_t> &local_size, bool async) const;

private:
    cl_command_queue queue_;
    cl_kernel kernel_;
};

Kernel::KernelImpl::KernelImpl(cl_command_queue queue, cl_kernel kernel) : queue_(queue), kernel_(kernel) {}

bool Kernel::KernelImpl::SetArg(cl_uint index, size_t size, const void *value) const
{
    cl_int ret = clSetKernelArg(kernel_, index, size, value);
    CHECK_OPENCL_ERROR_RETURN_FALSE(ret, "Failed to set kernel argument");
    return true;
}

bool Kernel::KernelImpl::Run(
    const std::vector<size_t> &global_size, const std::vector<size_t> &local_size, bool async) const
{
    cl_int ret = clEnqueueNDRangeKernel(
        queue_, kernel_, global_size.size(), nullptr, global_size.data(), local_size.data(), 0, nullptr, nullptr);
    CHECK_OPENCL_ERROR_RETURN_FALSE(ret, "Failed to enqueue kernel");
    if (async) {
        ret = clFinish(queue_);
        CHECK_OPENCL_ERROR_RETURN_FALSE(ret, "Failed to finish command queue");
    }
    return true;
}

Kernel::Kernel(KernelImpl *impl) { impl_.reset(impl); }

bool Kernel::SetArgImpl(uint32_t index, size_t size, const void *value) const
{
    if (impl_ == nullptr) {
        return false;
    }
    return impl_->SetArg(index, size, value);
}

bool Kernel::RunImpl(const std::vector<size_t> &global_size, const std::vector<size_t> &local_size, bool async) const
{
    if (impl_ == nullptr) {
        return false;
    }
    return impl_->Run(global_size, local_size, async);
}

class Buffer::BufferImpl final {
public:
    explicit BufferImpl(BufferManager *manager, cl_command_queue command_queue, size_t size);
    ~BufferImpl();
    BufferImpl() = delete;
    BufferImpl(const BufferImpl &) = delete;
    BufferImpl &operator=(const BufferImpl &) = delete;
    BufferImpl(BufferImpl &&) = delete;
    BufferImpl &operator=(BufferImpl &&) = delete;

    cl_mem GetClMem() const;
    void *GetHostPtr() const;
    size_t GetSize() const;

private:
    BufferManager *manager_;
    cl_command_queue command_queue_;
    cl_mem buffer_;
    size_t size_;
    void *host_ptr_;
};

Buffer::BufferImpl::BufferImpl(BufferManager *manager, cl_command_queue command_queue, size_t size)
    : manager_(manager), command_queue_(command_queue), size_(size)
{
    cl_int ret;
    buffer_ = manager_->Create(size);
    if (buffer_ == nullptr) {
        std::cout << "Failed to create buffer" << std::endl;
        return;
    }
    host_ptr_ = clEnqueueMapBuffer(
        command_queue_, buffer_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size_, 0, nullptr, nullptr, &ret);
    CHECK_OPENCL_ERROR_NO_RETURN(ret, "Failed to map buffer");
}

Buffer::BufferImpl::~BufferImpl()
{
    if (buffer_ == nullptr) {
        return;
    }
    if (host_ptr_ != nullptr) {
        cl_int ret = clEnqueueUnmapMemObject(command_queue_, buffer_, host_ptr_, 0, nullptr, nullptr);
        CHECK_OPENCL_ERROR_NO_RETURN(ret, "Failed to unmap buffer");
    }
    manager_->Release(buffer_);
}

cl_mem Buffer::BufferImpl::GetClMem() const { return buffer_; }

void *Buffer::BufferImpl::GetHostPtr() const { return host_ptr_; }

size_t Buffer::BufferImpl::GetSize() const { return size_; }

Buffer::Buffer(BufferImpl *impl) { impl_.reset(impl); }

cl_mem Buffer::GetClMem() const
{
    if (impl_ == nullptr) {
        return nullptr;
    }
    return impl_->GetClMem();
}

void *Buffer::GetHostPtr() const
{
    if (impl_ == nullptr) {
        return nullptr;
    }
    return impl_->GetHostPtr();
}

size_t Buffer::GetSize() const
{
    if (impl_ == nullptr) {
        return 0;
    }
    return impl_->GetSize();
}

class TinyOCL::TinyOCLImpl final {
public:
    TinyOCLImpl();
    ~TinyOCLImpl() = default;
    TinyOCLImpl(const TinyOCLImpl &) = delete;
    TinyOCLImpl &operator=(const TinyOCLImpl &) = delete;
    TinyOCLImpl(TinyOCLImpl &&) = delete;
    TinyOCLImpl &operator=(TinyOCLImpl &&) = delete;

    std::shared_ptr<Kernel> CreateKernel(const std::string &program_name,
        const std::string &kernel_name,
        const std::set<std::string> &build_options) const;

    std::shared_ptr<Buffer> CreateBuffer(size_t size) const;

private:
    bool Init();

    std::vector<cl_device_id> devices_;
    std::unique_ptr<_cl_context, decltype(&clReleaseContext)> context_{nullptr, clReleaseContext};
    std::unique_ptr<_cl_command_queue, decltype(&clReleaseCommandQueue)> command_queue_{nullptr, clReleaseCommandQueue};
    std::unique_ptr<ProgramManager> program_manager_;
    std::unique_ptr<BufferManager> buffer_manager_;
};

TinyOCL::TinyOCLImpl::TinyOCLImpl()
{
    if (!Init()) {
        std::cout << "Failed to initialize TinyOCL" << std::endl;
    }
}

bool TinyOCL::TinyOCLImpl::Init()
{
    cl_uint num_platforms;
    cl_int ret;
    ret = clGetPlatformIDs(0, nullptr, &num_platforms);
    CHECK_OPENCL_ERROR_RETURN_FALSE(ret, "Failed to get number of platforms");
    if (num_platforms == 0) {
        std::cout << "No OpenCL platforms found" << std::endl;
        return false;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    ret = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    CHECK_OPENCL_ERROR_RETURN_FALSE(ret, "Failed to get platform IDs");

    for (const auto &platform : platforms) {
        cl_uint num_devices;
        ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (num_devices == 0) {
            continue;
        }
        devices_.resize(num_devices);
        ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices_.data(), nullptr);
        CHECK_OPENCL_ERROR_RETURN_FALSE(ret, "Failed to get device IDs");
        std::cout << "GPU device found" << std::endl;
        context_.reset(clCreateContext(nullptr, num_devices, devices_.data(), nullptr, nullptr, &ret));
        CHECK_OPENCL_ERROR_RETURN_FALSE(ret, "Failed to create context");

        cl_command_queue_properties properties[] = {0};
        command_queue_.reset(clCreateCommandQueueWithProperties(context_.get(), devices_[0], properties, &ret));
        CHECK_OPENCL_ERROR_RETURN_FALSE(ret, "Failed to create command queue");

        program_manager_.reset(new (std::nothrow) ProgramManager(devices_[0], context_.get()));
        if (!program_manager_) {
            std::cout << "Failed to create ProgramManager" << std::endl;
            return false;
        }

        buffer_manager_.reset(new (std::nothrow) BufferManager(context_.get(), command_queue_.get()));
        if (!buffer_manager_) {
            std::cout << "Failed to create BufferManager" << std::endl;
            return false;
        }
        return true;
    }
    std::cout << "No GPU devices found" << std::endl;
    return false;
}

std::shared_ptr<Kernel> TinyOCL::TinyOCLImpl::CreateKernel(
    const std::string &program_name, const std::string &kernel_name, const std::set<std::string> &build_options) const
{
    if (!program_manager_) {
        return nullptr;
    }

    if (!program_manager_->BuildProgram(program_name, build_options)) {
        return nullptr;
    };
    cl_kernel kernel = program_manager_->GetKernel(program_name, kernel_name);
    if (!kernel) {
        return nullptr;
    }
    std::unique_ptr<Kernel::KernelImpl> kernel_impl(new (std::nothrow) Kernel::KernelImpl(command_queue_.get(), kernel));
    if (!kernel_impl) {
        return nullptr;
    }
    return std::make_shared<Kernel>(kernel_impl.release());
}

std::shared_ptr<Buffer> TinyOCL::TinyOCLImpl::CreateBuffer(size_t size) const
{
    if (!buffer_manager_) {
        return nullptr;
    }
    std::unique_ptr<Buffer::BufferImpl> buffer_impl(new (std::nothrow) Buffer::BufferImpl(buffer_manager_.get(),
        command_queue_.get(),
        size));
    if (!buffer_impl) {
        return nullptr;
    }
    return std::make_shared<Buffer>(buffer_impl.release());
}

TinyOCL &TinyOCL::GetInstance()
{
    static TinyOCL instance;
    return instance;
}

TinyOCL::TinyOCL() : impl_(std::make_unique<TinyOCLImpl>()) {}

std::shared_ptr<Kernel> TinyOCL::CreateKernel(
    const std::string &program_name, const std::string &kernel_name, const std::set<std::string> &build_options) const
{
    if (!impl_) {
        return nullptr;
    }
    return impl_->CreateKernel(program_name, kernel_name, build_options);
}

std::shared_ptr<Buffer> TinyOCL::CreateBuffer(size_t size) const
{
    if (!impl_) {
        return nullptr;
    }
    return impl_->CreateBuffer(size);
}

}  // namespace tinyocl