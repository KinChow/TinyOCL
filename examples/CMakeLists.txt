cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output)
set(TINYOCL_OUTPUT_DIR ${EXECUTABLE_OUTPUT_PATH})

find_package(OpenCL QUIET)

include_directories(${TINYOCL_OUTPUT_DIR}/include)
link_directories(${TINYOCL_OUTPUT_DIR})
add_executable(example ${CMAKE_CURRENT_SOURCE_DIR}/examples.cpp)
target_link_libraries(example ${PROJECT_NAME})
if (OpenCL_FOUND)
    target_link_libraries(example OpenCL::OpenCL)
endif()
