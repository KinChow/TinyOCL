cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME TinyOCL)
project(${PROJECT_NAME})

set(DEBUG_VERSION "DebugMode")
set(RELEASE_VERSION "ReleaseMode")
set(COMPILE_MODE ${DEBUG_VERSION})
# debug mode (project name + compile time)
if(${COMPILE_MODE} MATCHES ${DEBUG_VERSION})
    set(USE_RELEASE_MODE 0)
    set(CMAKE_BUILD_TYPE "Debug")
# release mode (project name + version)
else()
    set(USE_RELEASE_MODE 1)
    set(CMAKE_BUILD_TYPE "Release")
endif()

string(TIMESTAMP COMPILE_TIME %y%m%d_%H%M%S)
set(VERSION_MAJOR 2)
set(VERSION_MINOR 0)
set(VERSION_PATCH 0)
set(VERSION_TWEAK 0)

set(CMAKE_CXX_STANDARD 17)

set(OPENCL_HEADER_DIR ${PROJECT_SOURCE_DIR}/external/OpenCL-Headers)
set(TINYOCL_INTERNAL_HEADER_DIR ${PROJECT_SOURCE_DIR}/internal)
set(TINYOCL_INTERFACE_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src TINYOCL_SRC_DIR)

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(BUILD_LIBRARY ON)
else()
    set(BUILD_LIBRARY OFF)
endif()

find_package(OpenCL QUIET)
if (OpenCL_FOUND)
    message(STATUS "OpenCL found.")
    message(STATUS "OpenCL library dir: ${OpenCL_LIBRARIES}")
    message(STATUS "OpenCL include dir: ${OpenCL_INCLUDE_DIRS}")
else()
    message(STATUS "OpenCL not found.")
    message(STATUS "Use OpenCL Wrapper.")
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/opencl-wrapper)
endif()

if (BUILD_LIBRARY)
    add_definitions(-DCL_TARGET_OPENCL_VERSION=210)

    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output)
    include_directories(
        ${OPENCL_HEADER_DIR}
        ${TINYOCL_INTERNAL_HEADER_DIR}
        ${TINYOCL_INTERFACE_HEADER_DIR}
    )
    add_library(${PROJECT_NAME} STATIC ${TINYOCL_SRC_DIR})
    if (OpenCL_FOUND)
        target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)
    else()
        target_link_libraries(${PROJECT_NAME} OpenCLWrapper)
    endif()
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${TINYOCL_INTERFACE_HEADER_DIR} ${LIBRARY_OUTPUT_PATH}/include
    )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${OPENCL_HEADER_DIR}/CL ${LIBRARY_OUTPUT_PATH}/include/CL
    )

    option(ENABLE_EXAMPLE "Enable examples of TinyOCL." OFF)
    if (ENABLE_EXAMPLE)
        add_subdirectory(${PROJECT_SOURCE_DIR}/examples)
    endif()

    option(ENABLE_TEST "Enable test of TinyOCL." OFF)
    if (ENABLE_TEST)
        add_subdirectory(${PROJECT_SOURCE_DIR}/external/googletest)
        add_subdirectory(${PROJECT_SOURCE_DIR}/test)
    endif()
else()
    add_library(${PROJECT_NAME} INTERFACE)
    target_include_directories(${PROJECT_NAME} INTERFACE
        ${OPENCL_HEADER_DIR}
        ${TINYOCL_INTERNAL_HEADER_DIR}
        ${TINYOCL_INTERFACE_HEADER_DIR}
    )
    target_sources(${PROJECT_NAME} INTERFACE ${TINYOCL_SRC_DIR})
    target_compile_definitions(${PROJECT_NAME} INTERFACE -DCL_TARGET_OPENCL_VERSION=210)
    if (OpenCL_FOUND)
        target_link_libraries(${PROJECT_NAME} INTERFACE OpenCL::OpenCL)
    else()
        target_link_libraries(${PROJECT_NAME} INTERFACE OpenCLWrapper)
    endif()
endif()