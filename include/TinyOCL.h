/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-13 00:00:00 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-17 02:30:10
 */

#ifndef __TINYOCL_TINYOCL_H__
#define __TINYOCL_TINYOCL_H__

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <CL/cl.h>

namespace tinyocl {

class Kernel final {
public:
    class KernelImpl;
    explicit Kernel(KernelImpl *impl);
    ~Kernel() = default;
    Kernel() = delete;
    Kernel(const Kernel &) = delete;
    Kernel &operator=(const Kernel &) = delete;
    Kernel(Kernel &&) = delete;
    Kernel &operator=(Kernel &&) = delete;

    template <typename T, typename... Ts>
    bool Run(
        const std::vector<size_t> &global_size, const std::vector<size_t> &local_size, bool async, T arg, Ts... args) const
    {
        bool ret = SetArg(0, arg, args...);
        if (!ret) {
            return false;
        }
        return RunImpl(global_size, local_size, async);
    }

private:
    template <typename T, typename... Ts>
    bool SetArg(uint32_t index, T arg, Ts... args) const
    {
        bool ret = SetArgImpl(index, sizeof(T), &arg);
        if (!ret) {
            return false;
        }
        if constexpr (sizeof...(args) > 0) {
            return SetArg(index + 1, args...);
        }
        return true;
    }

    bool SetArgImpl(uint32_t index, size_t arg_size, const void *arg_value) const;

    bool RunImpl(const std::vector<size_t> &global_size, const std::vector<size_t> &local_size, bool async) const;

    std::unique_ptr<KernelImpl> impl_;
};

class Buffer final {
public:
    class BufferImpl;
    explicit Buffer(BufferImpl *impl);
    ~Buffer() = default;
    Buffer() = delete;
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&) = delete;
    Buffer &operator=(Buffer &&) = delete;

    cl_mem GetClMem() const;
    void *GetHostPtr() const;
    size_t GetSize() const;

private:
    std::unique_ptr<BufferImpl> impl_;
};

class TinyOCL final {
public:
    static TinyOCL &GetInstance();
    ~TinyOCL() = default;
    TinyOCL(const TinyOCL &) = delete;
    TinyOCL &operator=(const TinyOCL &) = delete;

    std::shared_ptr<Kernel> CreateKernel(const std::string &program_name, const std::string &kernel_name, const std::set<std::string> &build_options) const;

    std::shared_ptr<Buffer> CreateBuffer(size_t size) const;

private:
    TinyOCL();
    class TinyOCLImpl;
    std::unique_ptr<TinyOCLImpl> impl_;
};

}  // namespace tinyocl

#endif  // __TINYOCL_TINYOCL_H__
