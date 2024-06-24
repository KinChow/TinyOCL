/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-16 12:07:47 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-24 23:25:23
 */

#ifndef __TINYOCL_PROGRAMMANAGER_H__
#define __TINYOCL_PROGRAMMANAGER_H__

#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <CL/cl.h>

namespace TinyOCL {

struct ProgramWithKernels final {
    std::unique_ptr<_cl_program, decltype(&clReleaseProgram)> program{nullptr, clReleaseProgram};
    std::unordered_map<std::string, std::unique_ptr<_cl_kernel, decltype(&clReleaseKernel)>> kernels;
};

class ProgramManager final {
public:
    explicit ProgramManager(cl_device_id device, cl_context context);
    ~ProgramManager() = default;
    ProgramManager() = delete;
    ProgramManager(const ProgramManager &) = delete;
    ProgramManager &operator=(const ProgramManager &) = delete;
    ProgramManager(ProgramManager &&) = delete;
    ProgramManager &operator=(ProgramManager &&) = delete;

    bool BuildProgram(const std::string &program_name, const std::set<std::string> &build_options);
    cl_kernel GetKernel(const std::string &program_name, const std::string &kernel_name);

private:
    bool BuildProgramWithSource(const std::string &program_name, const std::string &build_options);
    bool BuildProgramWithBinary(const std::string &program_name, const std::string &build_options);
    bool PrintBuildLog(cl_program program);
    cl_device_id device_;
    cl_context context_;
    std::unordered_map<std::string, ProgramWithKernels> programs_with_kernels_;
    std::mutex mutex_;
};

}  // namespace TinyOCL

#endif  //__TINYOCL_PROGRAMMANAGER_H__