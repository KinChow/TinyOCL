/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-16 12:07:30 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-17 03:02:35
 */

#ifndef __TINYOCL_BUFFERMANAGER_H__
#define __TINYOCL_BUFFERMANAGER_H__

#include <mutex>
#include <unordered_set>
#include <CL/cl.h>

namespace TinyOCL {
class BufferManager final {
public:
    explicit BufferManager(cl_context context, cl_command_queue queue);
    ~BufferManager();
    BufferManager() = delete;
    BufferManager(const BufferManager &) = delete;
    BufferManager &operator=(const BufferManager &) = delete;
    BufferManager(BufferManager &&) = delete;
    BufferManager &operator=(BufferManager &&) = delete;

    cl_mem Create(size_t size);

    void Release(cl_mem buffer);

private:
    cl_context context_;
    cl_command_queue queue_;
    std::unordered_set<cl_mem> buffers_;
    std::mutex mutex_;
};

}  // namespace TinyOCL

#endif  //__TINYOCL_BUFFERMANAGER_H__