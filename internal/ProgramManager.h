/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-16 12:07:47 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-24 23:42:15
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
/**
 * @brief ProgramWithKernels is a class that manages OpenCL programs and kernels.
 * 
 */
struct ProgramWithKernels final {
    std::unique_ptr<_cl_program, decltype(&clReleaseProgram)> program{nullptr, clReleaseProgram};
    std::unordered_map<std::string, std::unique_ptr<_cl_kernel, decltype(&clReleaseKernel)>> kernels;
};

/**
 * @brief ProgramManager is a class that manages OpenCL programs.
 * 
 */
class ProgramManager final {
public:
    /**
     * @brief Construct a new Program Manager object
     * 
     * @param device OpenCL device
     * @param context OpenCL context
     */
    explicit ProgramManager(cl_device_id device, cl_context context);

    /**
     * @brief Destroy the Program Manager object
     * 
     */
    ~ProgramManager() = default;

    /**
     * @brief Delete default constructor
     * 
     */
    ProgramManager() = delete;

    /**
     * @brief Delete copy constructor
     * 
     */
    ProgramManager(const ProgramManager &) = delete;

    /**
     * @brief Delete copy assignment operator
     * 
     * @return ProgramManager& 
     */
    ProgramManager &operator=(const ProgramManager &) = delete;

    /**
     * @brief Delete move constructor
     * 
     */
    ProgramManager(ProgramManager &&) = delete;

    /**
     * @brief Delete move assignment operator
     * 
     * @return ProgramManager& 
     */
    ProgramManager &operator=(ProgramManager &&) = delete;

    /**
     * @brief Build a program
     * 
     * @param program_name
     * @param build_options 
     * @return true 
     * @return false 
     */
    bool BuildProgram(const std::string &program_name, const std::set<std::string> &build_options);

    /**
     * @brief Get a kernel
     * 
     * @param program_name 
     * @param kernel_name 
     * @return cl_kernel 
     */
    cl_kernel GetKernel(const std::string &program_name, const std::string &kernel_name);

private:
    /**
     * @brief Build a program with source
     * 
     * @param program_name 
     * @param build_options 
     * @return true 
     * @return false 
     */
    bool BuildProgramWithSource(const std::string &program_name, const std::string &build_options);

    /**
     * @brief Build a program with binary
     * 
     * @param program_name 
     * @param build_options 
     * @return true 
     * @return false 
     */
    bool BuildProgramWithBinary(const std::string &program_name, const std::string &build_options);

    /**
     * @brief Print the build log
     * 
     * @param program 
     * @return true 
     * @return false 
     */
    bool PrintBuildLog(cl_program program);

    cl_device_id device_;
    cl_context context_;
    std::unordered_map<std::string, ProgramWithKernels> programs_with_kernels_;
    std::mutex mutex_;
};

}  // namespace TinyOCL

#endif  //__TINYOCL_PROGRAMMANAGER_H__