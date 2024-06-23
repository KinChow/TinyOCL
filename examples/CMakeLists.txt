cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)

set(TINYOCL_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

find_package(OpenCL QUIET)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output)
include_directories(${TINYOCL_HEADER_DIR})
link_directories(${TINYOCL_HEADER_DIR})
add_executable(example ${CMAKE_CURRENT_SOURCE_DIR}/examples.cpp)
target_link_libraries(example ${PROJECT_NAME})
if (OpenCL_FOUND)
    target_link_libraries(example OpenCL::OpenCL)
endif()
