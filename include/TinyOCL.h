/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-13 00:00:00 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-17 23:11:25
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
    /**
     * @brief Implementation of Kernel
     *
     */
    class KernelImpl;

    /**
     * @brief Construct a new Kernel object
     * 
     * @param impl 
     */
    explicit Kernel(KernelImpl *impl);

    /**
     * @brief Destroy the Kernel object
     * 
     */
    ~Kernel() = default;

    /**
     * @brief Delete default constructor
     * 
     */
    Kernel() = delete;

    /**
     * @brief Delete copy constructor
     * 
     */
    Kernel(const Kernel &) = delete;

    /**
     * @brief Delete copy assignment operator
     * 
     * @return Kernel& 
     */
    Kernel &operator=(const Kernel &) = delete;

    /**
     * @brief Delete move constructor
     * 
     */
    Kernel(Kernel &&) = delete;

    /**
     * @brief Delete move assignment operator
     * 
     * @return Kernel& 
     */
    Kernel &operator=(Kernel &&) = delete;

    /**
     * @brief Run the kernel
     * @tparam T The type of the argument
     * @tparam Ts The types of the arguments
     * @param global_size The number of work items in each dimension
     * @param local_size The number of work items in each work group
     * @param async Whether to run the kernel asynchronously
     * @param arg The argument
     * @param args The arguments
     * @return true
     * @return false
     */
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
    /**
     * @brief Set the argument of the kernel
     *
     * @tparam T The type of the argument
     * @tparam Ts The types of the arguments
     * @param index The index of the argument
     * @param arg The argument
     * @param args The arguments
     * @return true
     * @return false
     */
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

    /**
     * @brief Set the argument of the kernel
     *
     * @param index The index of the argument
     * @param arg_size The size of the argument
     * @param arg_value The value of the argument
     * @return true
     * @return false
     */
    bool SetArgImpl(uint32_t index, size_t arg_size, const void *arg_value) const;

    /**
     * @brief Run the kernel
     *
     * @param global_size The number of work items in each dimension
     * @param local_size The number of work items in each work group
     * @param async Whether to run the kernel asynchronously
     * @return true
     * @return false
     */
    bool RunImpl(const std::vector<size_t> &global_size, const std::vector<size_t> &local_size, bool async) const;

    /**
     * @brief The pointer to the implementation of Kernel
     *
     */
    std::unique_ptr<KernelImpl> impl_;
};

/**
 * @brief MemcpyKind is an enum class that represents the kind of memcopy.
 *
 */
enum class MemcpyKind {
    HostToDevice,
    DeviceToHost,
};

class Buffer final {
public:
    /**
     * @brief Implementation of Buffer
     *
     */
    class BufferImpl;

    /**
     * @brief Construct a new Buffer object
     *
     * @param impl
     */
    explicit Buffer(BufferImpl *impl);

    /**
     * @brief Destroy the Buffer object
     *
     */
    ~Buffer() = default;

    /**
     * @brief Delete default constructor
     *
     */
    Buffer() = delete;

    /**
     * @brief Delete copy constructor
     *
     */
    Buffer(const Buffer &) = delete;

    /**
     * @brief Delete copy assignment operator
     *
     * @return Buffer&
     */
    Buffer &operator=(const Buffer &) = delete;

    /**
     * @brief Delete move constructor
     *
     */
    Buffer(Buffer &&) = delete;

    /**
     * @brief Delete move assignment operator
     *
     * @return Buffer&
     */
    Buffer &operator=(Buffer &&) = delete;

    /**
     * @brief Get the Cl Mem object
     * 
     * @return cl_mem 
     */
    cl_mem GetClMem() const;

    /**
     * @brief Get the Host Ptr
     * 
     * @return T* 
     */
    template <typename T>
    T *GetHostPtr() const
    {
        return static_cast<T *>(GetHostPtrImpl());
    }

    /**
     * @brief Get the buffer size
     * 
     * @return size_t 
     */
    size_t GetSize() const;

    /**
     * @brief Memcpy
     * 
     * @param host_ptr The host pointer
     * @param size The size of the memory to be copied
     * @param kind The kind of the memory copy
     * @return true 
     * @return false 
     */
    bool Memcpy(void *host_ptr, size_t size, MemcpyKind kind) const;

private:
    /**
     * @brief Get the host pointer
     * 
     * @return void* 
     */
    void *GetHostPtrImpl() const;

    /**
     * @brief The pointer to the implementation of Buffer
     *
     */
    std::unique_ptr<BufferImpl> impl_;
};

class TinyOCL final {
public:
    /**
     * @brief Get the Instance object
     * 
     * @return TinyOCL& 
     */
    static TinyOCL &GetInstance();

    /**
     * @brief Destroy the TinyOCL object
     * 
     */
    ~TinyOCL() = default;

    /**
     * @brief Delete copy constructor
     * 
     */
    TinyOCL(const TinyOCL &) = delete;

    /**
     * @brief Delete copy assignment operator
     * 
     * @return TinyOCL& 
     */
    TinyOCL &operator=(const TinyOCL &) = delete;

    /**
     * @brief Create a Kernel object
     * 
     * @param program_name The name of the program
     * @param kernel_name The name of the kernel
     * @param build_options The build options
     * @return std::shared_ptr<Kernel> 
     */
    std::shared_ptr<Kernel> CreateKernel(const std::string &program_name,
        const std::string &kernel_name,
        const std::set<std::string> &build_options) const;

    /**
     * @brief Create a Buffer object
     *
     * @param size The size of the buffer
     * @return std::shared_ptr<Buffer>
     */
    std::shared_ptr<Buffer> CreateBuffer(size_t size) const;

private:
    /** 
     * @brief Construct a new TinyOCL object
     * 
     */
    TinyOCL();

    /**
     * @brief The implementation of TinyOCL
     * 
     */
    class TinyOCLImpl;

    /**
     * @brief The pointer to the implementation of TinyOCL
     *
     */
    std::unique_ptr<TinyOCLImpl> impl_;
};

}  // namespace tinyocl

#endif  // __TINYOCL_TINYOCL_H__
