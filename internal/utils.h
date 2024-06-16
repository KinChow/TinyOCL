/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-16 15:52:56 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-16 15:57:36
 */

#ifndef __TINYOCL_UTILS_H__
#define __TINYOCL_UTILS_H__

#include <iostream>
#include <CL/cl.h>

#ifndef CHECK_OPENCL_ERROR
#define CHECK_OPENCL_ERROR(ret, msg, ret_val)                                        \
    do {                                                                             \
        if (ret != CL_SUCCESS) {                                                     \
            std::cout << "OpenCL error: " << msg << " (" << ret << ")" << std::endl; \
            return ret_val;                                                          \
        }                                                                            \
    } while (0)
#endif  // CHECK_OPENCL_ERROR

#ifndef CHECK_OPENCL_ERROR_NO_RETURN
#define CHECK_OPENCL_ERROR_NO_RETURN(ret, msg)                                       \
    do {                                                                             \
        if (ret != CL_SUCCESS) {                                                     \
            std::cout << "OpenCL error: " << msg << " (" << ret << ")" << std::endl; \
        }                                                                            \
    } while (0)
#endif  // CHECK_OPENCL_ERROR_NO_RETURN

#ifndef CHECK_OPENCL_ERROR_RETURN_FALSE
#define CHECK_OPENCL_ERROR_RETURN_FALSE(ret, msg) CHECK_OPENCL_ERROR(ret, msg, false)
#endif  // CHECK_OPENCL_ERROR_RETURN_FALSE

#ifndef CHECK_OPENCL_ERROR_RETURN_NULL
#define CHECK_OPENCL_ERROR_RETURN_NULL(ret, msg) CHECK_OPENCL_ERROR(ret, msg, nullptr)
#endif  // CHECK_OPENCL_ERROR_RETURN_NULL

#endif  // __TINYOCL_UTILS_H__