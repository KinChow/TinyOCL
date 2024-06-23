cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)

set(TINYOCL_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

include_directories(${TINYOCL_HEADER_DIR})

link_directories(${PROJECT_SOURCE_DIR}/output)

add_executable(example ${CMAKE_CURRENT_SOURCE_DIR}/examples.cpp)

target_link_libraries(example ${PROJECT_NAME} OpenCL::OpenCL)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output)