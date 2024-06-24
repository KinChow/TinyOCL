/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-16 12:07:30 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-24 23:42:34
 */

#ifndef __TINYOCL_BUFFERMANAGER_H__
#define __TINYOCL_BUFFERMANAGER_H__

#include <mutex>
#include <unordered_set>
#include <CL/cl.h>

namespace TinyOCL {
/**
 * @brief BufferManager is a class that manages OpenCL buffers.
 * 
 */
class BufferManager final {
public:
    /**
     * @brief Construct a new BufferManager object
     * 
     * @param context OpenCL context
     * @param queue OpenCL command queue
     */
    explicit BufferManager(cl_context context, cl_command_queue queue);

    /**
     * @brief Destroy the BufferManager object
     * 
     */
    ~BufferManager();

    /**
     * @brief Delete default constructor
     * 
     */
    BufferManager() = delete;

    /**
     * @brief Delete copy constructor
     * 
     */
    BufferManager(const BufferManager &) = delete;

    /**
     * @brief Delete copy assignment operator
     * 
     * @return BufferManager& 
     */
    BufferManager &operator=(const BufferManager &) = delete;

    /**
     * @brief Delete move constructor
     * 
     */
    BufferManager(BufferManager &&) = delete;

    /**
     * @brief Delete move assignment operator
     * 
     * @return BufferManager& 
     */
    BufferManager &operator=(BufferManager &&) = delete;

    /**
     * @brief Create a new buffer
     * 
     * @param size Buffer size
     * @return cl_mem 
     */
    cl_mem Create(size_t size);

    /**
     * @brief Release a buffer
     * 
     * @param buffer 
     */
    void Release(cl_mem buffer);

private:
    cl_context context_;
    cl_command_queue queue_;
    std::unordered_set<cl_mem> buffers_;
    std::mutex mutex_;
};

}  // namespace TinyOCL

#endif  //__TINYOCL_BUFFERMANAGER_H__