/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-16 14:16:47 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-16 17:16:47
 */

#include <fstream>
#include <iostream>
#include <regex>
#include "utils.h"
#include "ProgramManager.h"

namespace TinyOCL {

ProgramManager::ProgramManager(cl_device_id device, cl_context context) : device_(device), context_(context) {}

bool ProgramManager::BuildProgram(const std::string &program_name, const std::set<std::string> &build_options)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto program_iter = programs_with_kernels_.find(program_name);
    if (program_iter != programs_with_kernels_.end()) {
        return true;
    }
    std::string build_options_str = "";
    for (const std::string &option : build_options) {
        build_options_str += option + " ";
    }
    std::regex source_regex(R"(.*\.cl)");
    std::regex binary_regex(R"(.*\.bin)");
    if (std::regex_match(program_name, source_regex)) {
        return BuildProgramWithSource(program_name, build_options_str);
    } else if (std::regex_match(program_name, binary_regex)) {
        return BuildProgramWithBinary(program_name, build_options_str);
    } else {
        std::cout << "Invalid program name: " << program_name << std::endl;
        return false;
    }
}

bool ProgramManager::BuildProgramWithSource(const std::string &program_name, const std::string &build_options)
{
    std::ifstream program_file(program_name, std::ifstream::in | std::ifstream::binary);
    if (!program_file.is_open()) {
        std::cout << "Failed to open program file: " << program_name << std::endl;
        return false;
    }
    program_file.seekg(0, program_file.end);
    const size_t program_size = program_file.tellg();
    program_file.seekg(0, program_file.beg);
    std::vector<char> program_source(program_size + 1);
    program_source[program_size] = '\0';
    program_file.read(program_source.data(), program_size);
    program_file.close();

    cl_int ret;
    constexpr int num_programs = 1;
    const char *program_sources[num_programs] = {program_source.data()};
    const size_t program_sizes[num_programs] = {program_size};
    std::unique_ptr<_cl_program, decltype(&clReleaseProgram)> program(
        clCreateProgramWithSource(context_, 1, program_sources, program_sizes, &ret), clReleaseProgram);
    CHECK_OPENCL_ERROR_RETURN_FALSE(ret, "Failed to create program with source");
    ret = clBuildProgram(program.get(), 1, &device_, build_options.c_str(), nullptr, nullptr);
    if (ret != CL_SUCCESS) {
        std::cout << "Failed to build program: " << program_name << std::endl;
        PrintBuildLog(program.get());
        return false;
    }
    ProgramWithKernels program_with_kernels;
    program_with_kernels.program.reset(program.release());
    programs_with_kernels_.emplace(program_name, std::move(program_with_kernels));
    return true;
}

bool ProgramManager::BuildProgramWithBinary(const std::string &program_name, const std::string &build_options)
{
    std::ifstream program_file(program_name, std::ifstream::in | std::ifstream::binary);
    if (!program_file.is_open()) {
        std::cout << "Failed to open program file: " << program_name << std::endl;
        return false;
    }
    program_file.seekg(0, program_file.end);
    const size_t program_size = program_file.tellg();
    program_file.seekg(0, program_file.beg);
    std::vector<uint8_t> program_binary(program_size);
    program_file.read(reinterpret_cast<char *>(program_binary.data()), program_size);
    program_file.close();

    cl_int ret;
    constexpr int num_programs = 1;
    const uint8_t *program_binaries[num_programs] = {program_binary.data()};
    const size_t program_sizes[num_programs] = {program_size};
    std::unique_ptr<_cl_program, decltype(&clReleaseProgram)> program(
        clCreateProgramWithBinary(context_, 1, &device_, program_sizes, program_binaries, nullptr, &ret),
        clReleaseProgram);
    CHECK_OPENCL_ERROR_RETURN_FALSE(ret, "Failed to create program with binary");
    ret = clBuildProgram(program.get(), 1, &device_, build_options.c_str(), nullptr, nullptr);
    if (ret != CL_SUCCESS) {
        std::cout << "Failed to build program: " << program_name << std::endl;
        PrintBuildLog(program.get());
        return false;
    }
    ProgramWithKernels program_with_kernels;
    program_with_kernels.program.reset(program.release());
    programs_with_kernels_.emplace(program_name, std::move(program_with_kernels));
    return true;
}

bool ProgramManager::PrintBuildLog(cl_program program)
{
    size_t build_log_size;
    cl_int ret = clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &build_log_size);
    CHECK_OPENCL_ERROR_RETURN_FALSE(ret, "Failed to get program build log size");
    std::vector<char> build_log(build_log_size + 1);
    build_log[build_log_size] = '\0';
    ret = clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, build_log_size, build_log.data(), nullptr);
    CHECK_OPENCL_ERROR_RETURN_FALSE(ret, "Failed to get program build log");
    std::cout << "Build log: " << build_log.data() << std::endl;
    return true;
}

cl_kernel ProgramManager::GetKernel(const std::string &program_name, const std::string &kernel_name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto program_iter = programs_with_kernels_.find(program_name);
    if (program_iter == programs_with_kernels_.end()) {
        std::cout << "Program not found: " << program_name << std::endl;
        return nullptr;
    }
    auto kernel_iter = program_iter->second.kernels.find(kernel_name);
    if (kernel_iter != program_iter->second.kernels.end()) {
        return kernel_iter->second.get();
    }
    cl_int ret;
    program_iter->second.kernels.emplace(kernel_name,
        std::unique_ptr<_cl_kernel, decltype(&clReleaseKernel)>(
            clCreateKernel(program_iter->second.program.get(), kernel_name.c_str(), &ret), clReleaseKernel));
    CHECK_OPENCL_ERROR_RETURN_NULL(ret, "Failed to create kernel");
    return program_iter->second.kernels.at(kernel_name).get();
}

}  // namespace TinyOCL